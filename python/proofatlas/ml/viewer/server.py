"""
Training run viewer server.

A simple Flask server for viewing training runs with live updates.
Runs locally and auto-refreshes as training progresses.

Usage:
    proofatlas-viewer [--port 5000] [--log-dir .logs]
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from flask import Flask, render_template, jsonify, Response, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


def create_app(log_dir: str = ".logs") -> "Flask":
    """Create the Flask application."""
    if not FLASK_AVAILABLE:
        raise ImportError("Flask required. Install with: pip install flask")

    app = Flask(
        __name__,
        template_folder=Path(__file__).parent / "templates",
        static_folder=Path(__file__).parent / "static",
    )
    app.config["LOG_DIR"] = Path(log_dir)

    @app.route("/")
    def index():
        """Main page."""
        return render_template("index.html")

    @app.route("/api/runs")
    def list_runs():
        """List all training runs."""
        log_dir = app.config["LOG_DIR"]
        runs = []

        if log_dir.exists():
            for run_dir in sorted(log_dir.iterdir(), reverse=True):
                if run_dir.is_dir():
                    metrics_file = run_dir / "metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file) as f:
                                data = json.load(f)
                            runs.append({
                                "name": run_dir.name,
                                "start_time": data.get("start_time", ""),
                                "end_time": data.get("end_time"),
                                "num_epochs": len(data.get("epochs", [])),
                                "best_val_loss": data.get("best_val_loss"),
                                "model_type": data.get("config", {}).get("model", {}).get("type", "unknown"),
                                "status": "completed" if data.get("end_time") else "running",
                                "termination_reason": data.get("termination_reason"),
                            })
                        except (json.JSONDecodeError, IOError):
                            runs.append({
                                "name": run_dir.name,
                                "status": "error",
                            })

        return jsonify(runs)

    @app.route("/api/runs/<run_name>")
    def get_run(run_name: str):
        """Get full details for a specific run."""
        log_dir = app.config["LOG_DIR"]
        metrics_file = log_dir / run_name / "metrics.json"

        if not metrics_file.exists():
            return jsonify({"error": "Run not found"}), 404

        try:
            with open(metrics_file) as f:
                data = json.load(f)
            return jsonify(data)
        except (json.JSONDecodeError, IOError) as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/runs/<run_name>/stream")
    def stream_run(run_name: str):
        """Server-Sent Events stream for live updates."""
        log_dir = app.config["LOG_DIR"]
        metrics_file = log_dir / run_name / "metrics.json"

        def generate():
            last_mtime = 0
            last_epochs = 0

            while True:
                if metrics_file.exists():
                    mtime = metrics_file.stat().st_mtime
                    if mtime > last_mtime:
                        last_mtime = mtime
                        try:
                            with open(metrics_file) as f:
                                data = json.load(f)

                            num_epochs = len(data.get("epochs", []))
                            if num_epochs > last_epochs or data.get("end_time"):
                                last_epochs = num_epochs
                                yield f"data: {json.dumps(data)}\n\n"

                            # Stop streaming if run is complete
                            if data.get("end_time"):
                                break
                        except (json.JSONDecodeError, IOError):
                            pass

                time.sleep(1)  # Poll every second

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.route("/api/compare")
    def compare_runs():
        """Compare multiple runs."""
        run_names = request.args.getlist("runs")
        log_dir = app.config["LOG_DIR"]

        results = []
        for name in run_names:
            metrics_file = log_dir / name / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        data = json.load(f)
                    results.append({
                        "name": name,
                        "data": data,
                    })
                except (json.JSONDecodeError, IOError):
                    pass

        return jsonify(results)

    return app


def run_server(
    log_dir: str = ".logs",
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
):
    """Run the viewer server."""
    app = create_app(log_dir)

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print(f"Starting training viewer at http://{host}:{port}")
    print(f"Watching log directory: {Path(log_dir).absolute()}")
    print("Press Ctrl+C to stop")

    app.run(host=host, port=port, debug=debug, threaded=True)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Training run viewer")
    parser.add_argument("--log-dir", default=".logs", help="Log directory to watch")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    run_server(
        log_dir=args.log_dir,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
