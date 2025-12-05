#!/bin/bash
# Setup SPASS theorem prover
# Downloads and builds SPASS from source to .spass/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BASE_DIR/configs/spass.json"

# Read config
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

VERSION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['version'])")
SOURCE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['source'])")
TARGET_DIR="$BASE_DIR/.spass"

echo "SPASS Setup"
echo "==========="
echo "Version: $VERSION"
echo "Source:  $SOURCE"
echo "Target:  $TARGET_DIR"
echo ""

# Check if already installed
if [ -x "$TARGET_DIR/SPASS" ]; then
    echo "SPASS is already installed at $TARGET_DIR/SPASS"
    "$TARGET_DIR/SPASS" 2>&1 | head -3 || true
    echo ""
    echo "To reinstall, remove the directory first: rm -rf $TARGET_DIR"
    exit 0
fi

# Check for build dependencies
echo "Checking build dependencies..."
if ! command -v gcc &> /dev/null; then
    echo "Error: gcc is required to build SPASS"
    echo "Install with: sudo apt-get install build-essential"
    exit 1
fi

if ! command -v flex &> /dev/null; then
    echo "Error: flex is required to build SPASS"
    echo "Install with: sudo apt-get install flex"
    exit 1
fi

if ! command -v bison &> /dev/null; then
    echo "Error: bison is required to build SPASS"
    echo "Install with: sudo apt-get install bison"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Download
ARCHIVE="spass${VERSION//./}.tgz"
if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading SPASS v$VERSION..."
    curl -L -o "$ARCHIVE" "$SOURCE"
else
    echo "Archive already exists, skipping download"
fi

# Extract (extracts directly into current directory)
echo "Extracting..."
tar -xzf "$ARCHIVE"

# Build (files are extracted directly, no subdirectory)
echo "Building SPASS..."
make

# Verify binary was created
if [ ! -x "SPASS" ]; then
    echo "Error: Build failed - SPASS binary not found"
    exit 1
fi

# Clean up source files, keep only the binary
echo "Cleaning up build files..."
find . -maxdepth 1 -type f ! -name "SPASS" -delete

# Verify
echo ""
echo "SPASS v$VERSION installed successfully!"
./SPASS 2>&1 | head -3 || true
