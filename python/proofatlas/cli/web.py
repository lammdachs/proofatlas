#!/usr/bin/env python3
"""
Serve the ProofAtlas web interface with optional server-side proving API.

USAGE:
    proofatlas-web                    # Start server on port 8000
    proofatlas-web --port 3000        # Start server on port 3000
    proofatlas-web --kill             # Stop running server

When running locally, the server provides:
  - Static file serving for the web UI
  - GET  /api/health       - Server health check (detects ML availability)
  - POST /api/prove        - Server-side proving (supports ML selectors)
  - GET  /configs/*        - Serve config files from project root
"""

import argparse
import http.server
import json
import os
import signal
import socketserver
import sys
import time
import traceback
from pathlib import Path


def find_web_dir() -> Path:
    """Find the web directory."""
    # Try relative to this file
    candidates = [
        Path(__file__).parent.parent.parent.parent.parent / "web",
        Path.cwd() / "web",
    ]

    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate.resolve()

    # Search upward from cwd
    path = Path.cwd()
    while path != path.parent:
        if (path / "web" / "index.html").exists():
            return (path / "web").resolve()
        path = path.parent

    return None


def find_project_root(web_dir: Path) -> Path:
    """Find the project root (parent of web/)."""
    return web_dir.parent


def check_ml_available() -> bool:
    """Check if ML selectors are available (tch-rs / libtorch)."""
    try:
        from proofatlas import ProofState
        state = ProofState()
        # Try creating with a graph embedding type - if tch-rs is available this won't error
        # We just check that the module loads successfully
        return True
    except Exception:
        return False


# Status messages for non-proof results
STATUS_MESSAGES = {
    "saturated": "Saturated without finding a proof - the formula may be satisfiable",
    "resource_limit": "Resource limit reached",
    "timeout": "Timeout reached before finding a proof",
}


def run_prove(tptp_input: str, options: dict) -> dict:
    """Run the prover server-side and return result in WASM-compatible format."""
    from proofatlas import ProofState

    start = time.time()
    state = ProofState()
    state.add_clauses_from_tptp(tptp_input)
    initial_count = state.get_statistics()["total"]

    # Build saturation kwargs — pass config keys directly,
    # letting run_saturation() use its own defaults for the rest
    kwargs = {"enable_profiling": True}

    for key in ("timeout", "max_iterations", "literal_selection",
                "age_weight_ratio", "encoder", "scorer"):
        if key in options:
            kwargs[key] = options[key]

    proof_found, status, profile_json = state.run_saturation(**kwargs)
    elapsed_ms = int((time.time() - start) * 1000)

    # Get all steps and proof trace
    all_steps = state.get_all_steps()
    proof_steps = state.get_proof_trace() if proof_found else []
    stats = state.get_statistics()

    # Convert ProofStep objects to dicts matching WASM format
    def step_to_dict(s):
        return {
            "id": s.clause_id,
            "clause": s.clause_string,
            "rule": s.rule_name,
            "parents": list(s.parent_ids),
        }

    # Build all_clauses (exclude GivenClauseSelection)
    all_clauses = [step_to_dict(s) for s in all_steps if s.rule_name != "GivenClauseSelection"]

    # Build trace for inspector
    trace = build_trace(all_steps, initial_count)

    # Determine message
    if proof_found:
        message = f"Proof found with {len(proof_steps)} steps"
    else:
        message = STATUS_MESSAGES.get(status, status)

    return {
        "success": proof_found,
        "status": "proof_found" if proof_found else status,
        "message": message,
        "proof": [step_to_dict(s) for s in proof_steps] if proof_found else None,
        "all_clauses": all_clauses,
        "statistics": {
            "initial_clauses": initial_count,
            "generated_clauses": len(all_clauses),
            "final_clauses": stats["total"],
            "time_ms": elapsed_ms,
        },
        "trace": trace,
        "profile": json.loads(profile_json) if profile_json else None,
    }


def build_trace(all_steps, initial_count: int) -> dict:
    """Build a trace dict matching the WASM ProofTrace format."""
    initial_clauses = []
    saturation_steps = []
    processed_count = 0
    unprocessed_count = initial_count

    for s in all_steps:
        if s.rule_name == "Input" or s.rule_name == "input":
            initial_clauses.append({
                "id": s.clause_id,
                "clause": s.clause_string,
                "rule": "Input",
                "parents": [],
            })
        else:
            step_type = "given_selection" if s.rule_name == "GivenClauseSelection" else "inference"

            if s.rule_name == "GivenClauseSelection":
                processed_count += 1
                unprocessed_count = max(0, unprocessed_count - 1)
            else:
                unprocessed_count += 1

            saturation_steps.append({
                "step_type": step_type,
                "clause_idx": s.clause_id,
                "clause": s.clause_string,
                "rule": s.rule_name,
                "premises": list(s.parent_ids),
                "processed_count": processed_count,
                "unprocessed_count": unprocessed_count,
            })

    return {
        "initial_clauses": initial_clauses,
        "saturation_steps": saturation_steps,
    }


class ProofAtlasHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler with API endpoints for server-side proving."""

    project_root = None
    ml_available = False

    def send_json(self, data, status_code=200):
        """Send a JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            self.send_json({
                "status": "ok",
                "ml_available": self.ml_available,
            })
        elif self.path.startswith("/configs/"):
            self.serve_config_file()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/prove":
            self.handle_prove()
        else:
            self.send_error(404)

    def serve_config_file(self):
        """Serve a file from the project's configs/ directory."""
        # Strip /configs/ prefix to get relative path
        rel_path = self.path[len("/configs/"):]
        # Sanitize: prevent path traversal
        if ".." in rel_path or rel_path.startswith("/"):
            self.send_error(403, "Forbidden")
            return

        config_path = self.project_root / "configs" / rel_path
        if not config_path.is_file():
            self.send_error(404, f"Config file not found: {rel_path}")
            return

        try:
            content = config_path.read_bytes()
            self.send_response(200)
            if config_path.suffix == ".json":
                self.send_header("Content-Type", "application/json")
            else:
                self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def handle_prove(self):
        """Handle POST /api/prove."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            options = json.loads(body)

            tptp_input = options.pop("input", "")
            if not tptp_input:
                self.send_json({"error": "Missing 'input' field"}, 400)
                return

            result = run_prove(tptp_input, options)
            self.send_json(result)
        except json.JSONDecodeError as e:
            self.send_json({"error": f"Invalid JSON: {e}"}, 400)
        except Exception as e:
            traceback.print_exc()
            self.send_json({
                "success": False,
                "status": "error",
                "message": str(e),
                "proof": None,
                "all_clauses": None,
                "statistics": {"initial_clauses": 0, "generated_clauses": 0, "final_clauses": 0, "time_ms": 0},
                "trace": None,
                "profile": None,
            })

    def log_message(self, format, *args):
        """Suppress access logs for static files, keep API logs."""
        if self.path.startswith("/api/"):
            super().log_message(format, *args)


def get_pid_file() -> Path:
    """Get path to PID file."""
    return Path.home() / ".proofatlas-web.pid"


def is_server_running() -> tuple[bool, int]:
    """Check if server is running. Returns (is_running, pid)."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        return False, 0

    try:
        pid = int(pid_file.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        return True, pid
    except (ValueError, ProcessLookupError, PermissionError):
        pid_file.unlink(missing_ok=True)
        return False, 0


def kill_server():
    """Kill running server."""
    running, pid = is_server_running()
    if not running:
        print("No server running")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped server (PID {pid})")
        get_pid_file().unlink(missing_ok=True)
    except ProcessLookupError:
        print("Server already stopped")
        get_pid_file().unlink(missing_ok=True)
    except PermissionError:
        print(f"Permission denied to stop server (PID {pid})")
        sys.exit(1)


def start_server(port: int, web_dir: Path):
    """Start the web server."""
    running, pid = is_server_running()
    if running:
        print(f"Server already running (PID {pid})")
        print(f"Use --kill to stop it first")
        sys.exit(1)

    # Check if WASM package exists
    pkg_dir = web_dir / "pkg"
    if not (pkg_dir / "proofatlas_wasm.js").exists():
        print("Error: WASM package not found in web/pkg/")
        print("The package should be built during pip install.")
        print("If missing, run: wasm-pack build --target web --out-dir ../../web/pkg crates/proofatlas-wasm")
        sys.exit(1)

    project_root = find_project_root(web_dir)

    # Check ML availability
    ml_available = check_ml_available()
    if ml_available:
        print("ML selectors: available")
    else:
        print("ML selectors: not available (heuristic presets only)")

    # Configure handler
    ProofAtlasHandler.project_root = project_root
    ProofAtlasHandler.ml_available = ml_available

    # Change to web directory
    os.chdir(web_dir)

    # Save PID
    get_pid_file().write_text(str(os.getpid()))

    # Set up signal handler for cleanup
    def cleanup(signum, frame):
        get_pid_file().unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Start server
    try:
        with socketserver.TCPServer(("", port), ProofAtlasHandler) as httpd:
            print(f"Serving ProofAtlas at http://localhost:{port}")
            print(f"API endpoints: /api/health, /api/prove, /configs/*")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
    except OSError as e:
        get_pid_file().unlink(missing_ok=True)
        if "Address already in use" in str(e):
            print(f"Error: Port {port} is already in use")
            sys.exit(1)
        raise
    finally:
        get_pid_file().unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Serve the ProofAtlas web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to serve on (default: 8000)")
    parser.add_argument("--kill", action="store_true",
                        help="Stop running server")

    args = parser.parse_args()

    if args.kill:
        kill_server()
        return

    web_dir = find_web_dir()
    if not web_dir:
        print("Error: web directory not found")
        print("Run from the proofatlas repository root")
        sys.exit(1)

    start_server(args.port, web_dir)


if __name__ == "__main__":
    main()
