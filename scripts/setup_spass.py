#!/usr/bin/env python3
"""
Setup SPASS theorem prover.
Downloads and builds SPASS from source to .spass/

Requires: gcc, flex, bison (build from source)
Supports: Linux
"""

import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def check_command(cmd: str) -> bool:
    """Check if a command is available."""
    return shutil.which(cmd) is not None


def check_build_dependencies() -> list:
    """Check for required build tools. Returns list of missing tools."""
    required = ["gcc", "flex", "bison", "make"]
    missing = [cmd for cmd in required if not check_command(cmd)]
    return missing


def get_install_hint(missing: list) -> str:
    """Get install hints for missing build tools."""
    if check_command("apt-get"):
        pkgs = " ".join(missing)
        if "gcc" in missing:
            pkgs = pkgs.replace("gcc", "build-essential")
        return f"sudo apt-get install {pkgs}"
    elif check_command("dnf"):
        pkgs = " ".join(missing)
        if "gcc" in missing:
            pkgs = pkgs.replace("gcc", "gcc make")
        return f"sudo dnf install {pkgs}"
    elif check_command("pacman"):
        pkgs = " ".join(missing)
        if "gcc" in missing:
            pkgs = pkgs.replace("gcc", "base-devel")
        return f"sudo pacman -S {pkgs}"
    else:
        return f"Install: {', '.join(missing)}"


def download_progress(block_num: int, block_size: int, total_size: int):
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        kb = downloaded / 1024
        total_kb = total_size / 1024
        print(f"\r[{bar}] {percent:5.1f}% ({kb:.0f}/{total_kb:.0f} KB)", end="", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup SPASS theorem prover")
    parser.add_argument("--force", "-f", action="store_true", help="Force reinstall")
    args = parser.parse_args()

    root = get_project_root()
    config_path = root / "configs" / "spass.json"

    with open(config_path) as f:
        config = json.load(f)

    version = config["version"]
    source = config["source"]
    target_dir = root / ".spass"
    binary_path = target_dir / "SPASS"

    print("SPASS Setup")
    print("=" * 50)
    print(f"Version: {version}")
    print(f"Source:  {source}")
    print(f"Target:  {target_dir}")
    print()

    # Check if already installed
    if binary_path.exists() and not args.force:
        print(f"SPASS is already installed at {binary_path}")
        try:
            result = subprocess.run([str(binary_path)],
                                    capture_output=True, text=True, timeout=5)
            for line in (result.stdout + result.stderr).split('\n')[:3]:
                if line.strip():
                    print(line)
        except Exception:
            pass
        print()
        print("To reinstall, use --force or remove the directory:")
        print(f"  rm -rf {target_dir}")
        return

    # Check build dependencies
    print("Checking build dependencies...")
    missing = check_build_dependencies()
    if missing:
        print(f"Error: Missing build tools: {', '.join(missing)}")
        print()
        print("Install with:")
        print(f"  {get_install_hint(missing)}")
        sys.exit(1)
    print("  All dependencies found")

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download
    archive_name = f"spass{version.replace('.', '')}.tgz"
    archive_path = target_dir / archive_name
    if not archive_path.exists() or args.force:
        print(f"\nDownloading SPASS v{version}...")
        urllib.request.urlretrieve(source, archive_path, download_progress)
        print()  # newline after progress
    else:
        print("\nArchive already exists, skipping download")

    # Extract (SPASS extracts files directly, no subdirectory)
    print("Extracting...")
    with tarfile.open(archive_path, 'r:gz') as tf:
        tf.extractall(target_dir)

    # Build
    print("Building SPASS...")
    result = subprocess.run(
        ["make"],
        cwd=target_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Build failed!")
        print(result.stderr)
        sys.exit(1)

    # Verify binary was created
    if not binary_path.exists():
        print("Error: Build failed - SPASS binary not found")
        sys.exit(1)

    # Clean up source files, keep only the binary
    print("Cleaning up build files...")
    for f in target_dir.iterdir():
        if f.name != "SPASS" and f.is_file():
            f.unlink()

    # Verify
    print()
    print(f"SPASS v{version} installed successfully!")
    try:
        result = subprocess.run([str(binary_path)],
                                capture_output=True, text=True, timeout=5)
        for line in (result.stdout + result.stderr).split('\n')[:3]:
            if line.strip():
                print(line)
    except Exception:
        pass


if __name__ == "__main__":
    main()
