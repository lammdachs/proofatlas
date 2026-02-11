#!/usr/bin/env python3
"""
Setup Vampire theorem prover.
Downloads pre-built binary for the current platform to .vampire/

Supports: Linux (x86_64 and aarch64)
"""

import json
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def detect_platform() -> str:
    """Detect current platform and return config key."""
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    return f"linux-{arch}"


def download_progress(block_num: int, block_size: int, total_size: int):
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r[{bar}] {percent:5.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup Vampire theorem prover")
    parser.add_argument("--force", "-f", action="store_true", help="Force reinstall")
    args = parser.parse_args()

    root = get_project_root()
    config_path = root / "configs" / "vampire.json"

    with open(config_path) as f:
        config = json.load(f)

    version = config["version"]
    plat = detect_platform()
    sources = config["sources"]

    if plat not in sources:
        print(f"Error: No Vampire binary available for platform: {plat}")
        print("Available platforms:")
        for k in sources:
            print(f"  - {k}")
        sys.exit(1)

    source = sources[plat]
    target_dir = root / ".vampire"

    binary_name = "vampire"
    binary_path = target_dir / binary_name

    print("Vampire Setup")
    print("=" * 50)
    print(f"Version:  {version}")
    print(f"Platform: {plat}")
    print(f"Source:   {source}")
    print(f"Target:   {target_dir}")
    print()

    # Check if already installed
    if binary_path.exists() and not args.force:
        print(f"Vampire is already installed at {binary_path}")
        try:
            result = subprocess.run([str(binary_path), "--version"],
                                    capture_output=True, text=True, timeout=5)
            if result.stdout:
                print(result.stdout.split('\n')[0])
        except Exception:
            pass
        print()
        print("To reinstall, use --force or remove the directory:")
        print(f"  rm -rf {target_dir}")
        return

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download
    archive_path = target_dir / f"vampire-{plat}.zip"
    if not archive_path.exists() or args.force:
        print(f"Downloading Vampire v{version} for {plat}...")
        urllib.request.urlretrieve(source, archive_path, download_progress)
        print()  # newline after progress
    else:
        print("Archive already exists, skipping download")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(archive_path, 'r') as zf:
        zf.extractall(target_dir)

    # Find and rename binary (releases have versioned names like vampire_z3_rel...)
    for f in target_dir.iterdir():
        if f.is_file() and f.name.startswith("vampire") and f.suffix != ".zip":
            if f.name != binary_name:
                target = target_dir / binary_name
                if target.exists():
                    target.unlink()
                f.rename(target)
            break

    binary_path.chmod(0o755)

    # Verify
    if binary_path.exists():
        print()
        print(f"Vampire v{version} installed successfully!")
        try:
            result = subprocess.run([str(binary_path), "--version"],
                                    capture_output=True, text=True, timeout=5)
            if result.stdout:
                print(result.stdout.split('\n')[0])
        except Exception:
            pass
    else:
        print("Error: Installation verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
