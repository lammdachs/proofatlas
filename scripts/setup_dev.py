#!/usr/bin/env python3
"""One-command development environment setup.

Runs all setup scripts that were previously called by setup.py during pip install.
After this, use `maturin develop --release` to build the Python extension.

Usage:
    python scripts/setup_dev.py           # Run all setup steps
    python scripts/setup_dev.py --skip-external  # Skip TPTP/Vampire/SPASS downloads
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_script(script: Path, description: str) -> bool:
    """Run a setup script, returning True on success."""
    if not script.exists():
        print(f"  Warning: {script} not found, skipping")
        return False

    print(f"  Running {script.name}...")
    try:
        subprocess.run([sys.executable, str(script)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Warning: {description} failed: {e}")
        print(f"  Run manually with: python {script}")
        return False


def setup_cargo_config():
    """Configure .cargo/config.toml for tch-rs/libtorch linking."""
    print("[1/3] Cargo config (libtorch rpath)")
    script = PROJECT_ROOT / "scripts" / "setup_cargo.py"
    run_script(script, "Cargo config")


def setup_external_tools():
    """Download TPTP, Vampire, and SPASS."""
    print("[2/3] External tools")

    # TPTP (all platforms)
    metadata = PROJECT_ROOT / ".data" / "problem_metadata.json"
    if metadata.exists():
        print("  TPTP already installed")
    else:
        run_script(PROJECT_ROOT / "scripts" / "setup_tptp.py", "TPTP setup")

    # Vampire and SPASS (Linux only)
    if platform.system().lower() == "linux":
        vampire = PROJECT_ROOT / ".vampire" / "vampire"
        if vampire.exists():
            print("  Vampire already installed")
        else:
            run_script(PROJECT_ROOT / "scripts" / "setup_vampire.py", "Vampire setup")

        spass = PROJECT_ROOT / ".spass" / "SPASS"
        if spass.exists():
            print("  SPASS already installed")
        else:
            run_script(PROJECT_ROOT / "scripts" / "setup_spass.py", "SPASS setup")
    else:
        print("  Skipping Vampire/SPASS (Linux only)")


def build_wasm():
    """Build the WASM package for the web interface."""
    print("[3/3] WASM package")

    wasm_crate = PROJECT_ROOT / "crates" / "proofatlas-wasm"
    web_pkg = PROJECT_ROOT / "web" / "pkg"

    if not (wasm_crate / "Cargo.toml").exists():
        print("  Warning: proofatlas-wasm crate not found, skipping")
        return

    if (web_pkg / "proofatlas_wasm.js").exists():
        print("  WASM package already built")
        return

    if not shutil.which("wasm-pack"):
        print("  Installing wasm-pack...")
        try:
            subprocess.run(
                ["cargo", "install", "wasm-pack"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to install wasm-pack: {e}")
            print("  Install manually with: cargo install wasm-pack")
            return

    print("  Building WASM package...")
    try:
        subprocess.run(
            ["wasm-pack", "build", "--target", "web", "--out-dir", str(web_pkg)],
            cwd=wasm_crate,
            check=True,
        )
        print(f"  WASM package built: {web_pkg}")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: WASM build failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Set up ProofAtlas development environment")
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip downloading TPTP, Vampire, and SPASS",
    )
    args = parser.parse_args()

    print("Setting up ProofAtlas development environment\n")

    # Check for Rust toolchain
    if not shutil.which("cargo"):
        print("Error: Rust toolchain not found.", file=sys.stderr)
        print("Install from https://rustup.rs/", file=sys.stderr)
        sys.exit(1)

    setup_cargo_config()

    if not args.skip_external:
        setup_external_tools()
    else:
        print("[2/3] External tools (skipped)")

    build_wasm()

    print("\nDone. Next steps:")
    print("  maturin develop --release    # Build and install the extension")
    print("  cargo test                   # Run Rust tests")
    print("  python -m pytest             # Run Python tests")


if __name__ == "__main__":
    main()
