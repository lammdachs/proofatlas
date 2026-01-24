#!/usr/bin/env python3
"""
Serve the ProofAtlas web interface.

USAGE:
    proofatlas-web                    # Start server on port 8000
    proofatlas-web --port 3000        # Start server on port 3000
    proofatlas-web --kill             # Stop running server
"""

import argparse
import http.server
import os
import signal
import socketserver
import sys
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
    handler = http.server.SimpleHTTPRequestHandler

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Serving ProofAtlas at http://localhost:{port}")
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
