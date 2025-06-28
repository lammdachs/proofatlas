#!/usr/bin/env python3
"""
Download and setup TPTP library for ProofAtlas.

This module provides functionality to download and extract the TPTP
(Thousands of Problems for Theorem Provers) library.
"""

import os
import sys
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple
import re
import shutil


def find_tptp_version() -> str:
    """Find the latest available TPTP version from the official website."""
    base_url = "https://tptp.org/TPTP/Distribution"
    
    try:
        # Fetch directory listing
        with urllib.request.urlopen(f"{base_url}/") as response:
            html = response.read().decode('utf-8')
        
        # Extract TPTP versions using regex
        pattern = r'href="TPTP-v(\d+\.\d+\.\d+)\.tgz"'
        versions = re.findall(pattern, html)
        
        if versions:
            # Sort versions and get the latest
            versions.sort(key=lambda v: tuple(map(int, v.split('.'))))
            return f"v{versions[-1]}"
        else:
            # Fallback if parsing fails
            return "v9.0.0"
    except Exception as e:
        print(f"Warning: Could not fetch TPTP versions: {e}")
        return "v9.0.0"


def check_existing_tptp(tptp_path: Path) -> Optional[str]:
    """Check for existing TPTP installation and return version if found."""
    if not tptp_path.exists():
        return None
    
    # Look for TPTP-v* directories
    for dir_path in tptp_path.glob("TPTP-v*"):
        if dir_path.is_dir() and (dir_path / "Problems").exists():
            # Extract version from directory name
            match = re.search(r'v(\d+\.\d+\.\d+)', dir_path.name)
            if match:
                return f"v{match.group(1)}"
    
    return None


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", 
                              end='', flush=True)
            
            print()  # New line after progress
            print("Download complete!")
            
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download file: {e}")


def extract_tptp_archive(archive_path: Path, extract_to: Path) -> str:
    """Extract TPTP archive and return the extracted directory name."""
    print("Extracting TPTP archive...")
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        # Get the root directory name from the archive
        root_dir = None
        for member in tar.getmembers():
            if '/' in member.name:
                root_dir = member.name.split('/')[0]
                break
        
        if not root_dir:
            raise RuntimeError("Could not determine TPTP root directory from archive")
        
        # Extract all files
        tar.extractall(path=extract_to)
    
    print(f"Extraction complete!")
    return root_dir


def setup_tptp(data_dir: Optional[Path] = None, force_download: bool = False) -> Tuple[bool, str]:
    """
    Download and setup TPTP library.
    
    Args:
        data_dir: Directory to store TPTP data. If None, uses .data/problems/tptp
        force_download: Force download even if TPTP already exists
    
    Returns:
        Tuple of (success, message)
    """
    # Determine TPTP directory
    if data_dir is None:
        # Use default location relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / ".data" / "problems" / "tptp"
    
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing installation
    existing_version = check_existing_tptp(data_dir)
    
    if existing_version and not force_download:
        return True, f"TPTP {existing_version} already installed at {data_dir}"
    
    # Find latest version
    latest_version = find_tptp_version()
    
    if existing_version == latest_version and not force_download:
        return True, f"TPTP {latest_version} (latest) already installed at {data_dir}"
    
    # Download TPTP
    archive_name = f"TPTP-{latest_version}.tgz"
    archive_path = data_dir / archive_name
    url = f"https://tptp.org/TPTP/Distribution/{archive_name}"
    
    try:
        # Download if archive doesn't exist
        if not archive_path.exists():
            download_file(url, archive_path)
        else:
            print(f"Using existing archive: {archive_path}")
        
        # Extract archive
        extracted_dir = extract_tptp_archive(archive_path, data_dir)
        
        # Verify extraction
        tptp_root = data_dir / extracted_dir
        if not (tptp_root / "Problems").exists():
            return False, f"Problems directory not found in extracted TPTP at {tptp_root}"
        
        # Optionally remove archive
        archive_path.unlink()
        
        message = f"""TPTP {latest_version} installed successfully!
TPTP root: {tptp_root}
Problems directory: {tptp_root / "Problems"}
Axioms directory: {tptp_root / "Axioms"}"""
        
        return True, message
        
    except Exception as e:
        return False, f"Failed to setup TPTP: {e}"


def main():
    """Command-line interface for TPTP download."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and setup TPTP library for ProofAtlas"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory to store TPTP data (default: .data/problems/tptp)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if TPTP already exists"
    )
    
    args = parser.parse_args()
    
    print("ProofAtlas TPTP Downloader")
    print("=" * 50)
    print()
    
    success, message = setup_tptp(args.data_dir, args.force)
    print(message)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()