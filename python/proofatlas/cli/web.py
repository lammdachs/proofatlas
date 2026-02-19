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
  - GET  /api/tptp/{name}  - Load TPTP problem by name (e.g., GRP001-1)
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

from proofatlas.paths import find_project_root


def find_web_dir(project_root: Path) -> Path:
    """Find the web directory to serve from, starting at the project root.

    Looks for SvelteKit build output (web/build/index.html) first,
    then falls back to legacy layout (web/index.html).
    """
    web = project_root / "web"
    if (web / "build" / "index.html").exists():
        return (web / "build").resolve()
    if (web / "index.html").exists():
        return web.resolve()
    return None


def check_ml_available() -> bool:
    """Check if ML selectors are available (tch-rs / libtorch)."""
    try:
        from proofatlas import ProofAtlas
        ProofAtlas()
        return True
    except Exception:
        return False


def run_prove(tptp_input: str, options: dict, tptp_root: str = None,
              project_root: Path = None) -> dict:
    """Run the prover server-side and return result in WASM-compatible format.

    Args:
        tptp_input: TPTP problem content
        options: Prover configuration options
        tptp_root: Optional path to TPTP root directory for resolving include() directives
        project_root: Project root directory for resolving weights path
    """
    from proofatlas import ProofAtlas

    # Build ProofAtlas orchestrator with all configuration
    atlas_kwargs = {"enable_profiling": True}
    if tptp_root:
        atlas_kwargs["include_dir"] = tptp_root

    for key in ("timeout", "max_iterations", "literal_selection",
                "age_weight_ratio", "encoder", "scorer"):
        if key in options:
            atlas_kwargs[key] = options[key]

    memory_limit = options.get("memory_limit")
    if memory_limit is not None:
        atlas_kwargs["memory_limit"] = memory_limit

    # Resolve weights path relative to project root (CWD may differ)
    if project_root and options.get("encoder"):
        weights_dir = project_root / ".weights"
        if weights_dir.exists():
            atlas_kwargs["weights_path"] = str(weights_dir)

    start = time.time()
    atlas = ProofAtlas(**atlas_kwargs)
    prover = atlas.prove_string(tptp_input)
    elapsed_ms = int((time.time() - start) * 1000)

    # Build the complete result in Rust (shared format with WASM)
    return json.loads(prover.prove_result_json(elapsed_ms))


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

    def end_headers(self):
        # Prevent caching for WASM/JS pkg files (rebuilt frequently during development)
        if self.path.startswith("/pkg/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            self.send_json({
                "status": "ok",
                "ml_available": self.ml_available,
            })
        elif self.path.startswith("/api/tptp/"):
            self.serve_tptp_problem()
        elif self.path.startswith("/configs/"):
            self.serve_config_file()
        else:
            super().do_GET()

    def serve_tptp_problem(self):
        """Serve a TPTP problem from the .tptp directory."""
        import urllib.parse
        # Extract problem name from path (e.g., /api/tptp/GRP001-1 -> GRP001-1)
        problem_name = urllib.parse.unquote(self.path[len("/api/tptp/"):])

        # Sanitize: prevent path traversal
        if ".." in problem_name or "/" in problem_name or "\\" in problem_name:
            self.send_json({"error": "Invalid problem name"}, 400)
            return

        # Find the TPTP directory
        tptp_dir = self.project_root / ".tptp"
        if not tptp_dir.exists():
            self.send_json({"error": "TPTP library not found. Run: uv run scripts/setup_tptp.py"}, 404)
            return

        # Look for the problem in TPTP directory structure
        # Problems are organized as: Problems/{DOMAIN}/{PROBLEM}.p
        # e.g., Problems/GRP/GRP001-1.p
        domain = problem_name[:3] if len(problem_name) >= 3 else problem_name

        # Try common TPTP version directories
        for version_dir in sorted(tptp_dir.iterdir(), reverse=True):
            if not version_dir.is_dir():
                continue
            problem_path = version_dir / "Problems" / domain / f"{problem_name}.p"
            if problem_path.exists():
                try:
                    content = problem_path.read_text()
                    self.send_json({"content": content})
                    return
                except Exception as e:
                    self.send_json({"error": f"Error reading problem: {e}"}, 500)
                    return

        self.send_json({"error": f"Problem not found: {problem_name}"}, 404)

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

            # Find TPTP root for resolving include() directives
            tptp_root = None
            tptp_dir = self.project_root / ".tptp"
            if tptp_dir.exists():
                # Find the TPTP version directory (e.g., TPTP-v9.0.0)
                for version_dir in sorted(tptp_dir.iterdir(), reverse=True):
                    if version_dir.is_dir() and version_dir.name.startswith("TPTP"):
                        tptp_root = str(version_dir)
                        break

            result = run_prove(tptp_input, options, tptp_root=tptp_root,
                               project_root=self.project_root)
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


def start_server(port: int, web_dir: Path, project_root: Path):
    """Start the web server."""
    running, pid = is_server_running()
    if running:
        print(f"Server already running (PID {pid})")
        print(f"Use --kill to stop it first")
        sys.exit(1)

    # Check if WASM package exists (in served directory's pkg/)
    pkg_dir = web_dir / "pkg"
    if not (pkg_dir / "proofatlas_wasm.js").exists():
        print("Warning: WASM package not found in pkg/")
        print("Browser-only proving will not work.")
        print("Build WASM: wasm-pack build --target web --out-dir ../../web/static/pkg crates/proofatlas-wasm")
        print("Then rebuild: cd web && npm run build")

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
            print(f"API endpoints: /api/health, /api/tptp/{{name}}, /api/prove, /configs/*")
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

    try:
        project_root = find_project_root()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    web_dir = find_web_dir(project_root)
    if not web_dir:
        print("Error: web directory not found")
        print(f"Expected web/build/index.html or web/index.html under {project_root}")
        sys.exit(1)

    start_server(args.port, web_dir, project_root)


if __name__ == "__main__":
    main()
