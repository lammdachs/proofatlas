#!/usr/bin/env python3
"""Single entry point for setting up ProofAtlas.

Usage:
    python scripts/setup.py                  # Full setup
    python scripts/setup.py --skip-external  # Skip TPTP/Vampire/SPASS
    python scripts/setup.py --skip-wasm      # Skip WASM build
    python scripts/setup.py --cpu            # Force CPU-only torch
"""

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_pkg_manager() -> str:
    """Detect the package manager for the active environment."""
    if shutil.which("uv"):
        return "uv"
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return "conda"
    return "pip"


def pkg_install(*packages: str, index_url: str | None = None):
    """Install packages using the detected package manager."""
    mgr = detect_pkg_manager()
    if mgr == "uv":
        cmd = ["uv", "pip", "install", *packages]
        if index_url:
            cmd += ["--index-url", index_url]
    elif mgr == "conda":
        cmd = ["conda", "install", "-y", *packages]
        # conda doesn't support --index-url; torch CUDA is handled via
        # pytorch channel, but for simplicity we fall back to pip for torch
        # when a custom index is needed.
        if index_url:
            cmd = [sys.executable, "-m", "pip", "install", *packages,
                   "--index-url", index_url]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *packages]
        if index_url:
            cmd += ["--index-url", index_url]
    subprocess.run(cmd, check=True)


def is_importable(module: str) -> bool:
    """Check whether a Python module can be imported."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def detect_cuda_version() -> str | None:
    """Parse CUDA version from nvidia-smi output, e.g. '12.4' -> 'cu124'."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10,
        )
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if match:
            major, minor = match.group(1), match.group(2)
            return f"cu{major}{minor}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


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


# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------

def check_venv():
    """Exit if no virtual environment is active."""
    if sys.prefix == sys.base_prefix:
        print("Error: No virtual environment detected.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Create and activate one first:", file=sys.stderr)
        print("  pip:   python -m venv .venv && source .venv/bin/activate", file=sys.stderr)
        print("  uv:    uv venv .venv && source .venv/bin/activate", file=sys.stderr)
        print("  conda: conda create -n proofatlas python=3.12 && conda activate proofatlas", file=sys.stderr)
        sys.exit(1)


def check_rust():
    """Check for Rust toolchain, offering to install if missing."""
    if shutil.which("cargo"):
        return

    print("  Rust toolchain not found.")
    answer = input("  Install via rustup? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        print("  Installing Rust...")
        subprocess.run(
            ["sh", "-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"],
            check=True,
        )
        # Add cargo to PATH for the rest of this session
        cargo_bin = Path.home() / ".cargo" / "bin"
        os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ['PATH']}"
        if not shutil.which("cargo"):
            print("Error: cargo still not found after install.", file=sys.stderr)
            sys.exit(1)
        print("  Rust installed successfully")
    else:
        print("Install Rust manually from https://rustup.rs/", file=sys.stderr)
        sys.exit(1)


def install_prerequisites(cpu: bool):
    """Install Python prerequisites in order."""
    mgr = detect_pkg_manager()
    print(f"  Package manager: {mgr}")

    # 1. maturin
    if not is_importable("maturin"):
        print("  Installing maturin...")
        pkg_install("maturin[patchelf]")
    else:
        print("  maturin already installed")

    # 2. numpy (needed before torch in setup_libtorch)
    if not is_importable("numpy"):
        print("  Installing numpy...")
        pkg_install("numpy")
    else:
        print("  numpy already installed")

    # 3. torch
    if not is_importable("torch"):
        print("  Installing torch...")
        if cpu:
            index_url = "https://download.pytorch.org/whl/cpu"
        else:
            cuda_tag = detect_cuda_version()
            if cuda_tag:
                index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
                print(f"  Detected CUDA: {cuda_tag}")
            else:
                index_url = None
                print("  No CUDA detected, installing default torch")
        pkg_install("torch>=2.9", index_url=index_url)
    else:
        print("  torch already installed")

    # 4. tqdm, pytest
    missing = [p for p in ("tqdm", "pytest") if not is_importable(p)]
    if missing:
        print(f"  Installing {', '.join(missing)}...")
        pkg_install(*missing)
    else:
        print("  tqdm, pytest already installed")


def build_extension():
    """Build the Rust extension with maturin."""
    print("  Running maturin develop...")
    env = os.environ.copy()
    env["LIBTORCH_USE_PYTORCH"] = "1"
    subprocess.run(["maturin", "develop"], cwd=PROJECT_ROOT, check=True, env=env)


def setup_external_tools():
    """Download TPTP, Vampire, and SPASS."""
    # TPTP (all platforms)
    metadata = PROJECT_ROOT / ".data" / "problem_metadata.json"
    if metadata.exists():
        print("  TPTP already installed")
    else:
        run_script(PROJECT_ROOT / "scripts" / "setup_tptp.py", "TPTP setup")

    # Vampire and SPASS
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


def build_wasm():
    """Build the WASM package for the web interface."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Set up ProofAtlas")
    parser.add_argument(
        "--skip-external", action="store_true",
        help="Skip downloading TPTP, Vampire, and SPASS",
    )
    parser.add_argument(
        "--skip-wasm", action="store_true",
        help="Skip building the WASM package",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU-only PyTorch (skip CUDA auto-detection)",
    )
    args = parser.parse_args()

    print("Setting up ProofAtlas\n")

    # Step 1: Virtual environment
    print("[1/5] Virtual environment")
    check_venv()
    print("  OK\n")

    # Step 2: Rust toolchain
    print("[2/5] Rust toolchain")
    check_rust()
    print("  OK\n")

    # Step 3: Python prerequisites
    print("[3/5] Python prerequisites")
    install_prerequisites(cpu=args.cpu)
    print()

    # Step 4: Build extension
    print("[4/5] Build extension")
    build_extension()
    print()

    # Step 5: External tools
    if not args.skip_external:
        print("[5/5] External tools")
        setup_external_tools()
    else:
        print("[5/5] External tools (skipped)")
    print()

    # Optional: MiniLM for trace embedding
    print("[ML] Base MiniLM (for trace embedding)")
    run_script(PROJECT_ROOT / "scripts" / "setup_minilm.py", "MiniLM setup")
    print()

    # Optional: WASM
    if not args.skip_wasm:
        print("[WASM] Web package")
        build_wasm()
        print()

    print("Done. Verify with:")
    print("  proofatlas --list")
    print("  cargo test -p proofatlas --lib")


if __name__ == "__main__":
    main()
