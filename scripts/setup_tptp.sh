#!/bin/bash
# Setup TPTP library
# Downloads and extracts TPTP to .tptp/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BASE_DIR/configs/tptp.json"

# Read config
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

VERSION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['version'])")
SOURCE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['source'])")
TARGET_DIR="$BASE_DIR/.tptp"

echo "TPTP Setup"
echo "=========="
echo "Version: $VERSION"
echo "Source:  $SOURCE"
echo "Target:  $TARGET_DIR"
echo ""

# Check if already installed
if [ -d "$TARGET_DIR/TPTP-v$VERSION" ]; then
    echo "TPTP v$VERSION is already installed at $TARGET_DIR/TPTP-v$VERSION"
    echo "To reinstall, remove the directory first: rm -rf $TARGET_DIR"
    exit 0
fi

# Create target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Download
ARCHIVE="TPTP-v$VERSION.tgz"
if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading TPTP v$VERSION..."
    curl -L -o "$ARCHIVE" "$SOURCE"
else
    echo "Archive already exists, skipping download"
fi

# Extract
echo "Extracting..."
tar -xzf "$ARCHIVE"

# Verify
if [ -d "TPTP-v$VERSION/Problems" ]; then
    echo ""
    echo "TPTP v$VERSION installed successfully!"
    echo "Problems directory: $TARGET_DIR/TPTP-v$VERSION/Problems"

    # Count problems
    PROBLEM_COUNT=$(find "TPTP-v$VERSION/Problems" -name "*.p" | wc -l)
    echo "Total problems: $PROBLEM_COUNT"
else
    echo "Error: Installation verification failed"
    exit 1
fi

# Extract problem metadata
echo ""
echo "Extracting problem metadata..."
cd "$BASE_DIR"
python3 "$SCRIPT_DIR/extract_problem_metadata.py"
echo "Problem metadata saved to .data/problem_metadata.json"
