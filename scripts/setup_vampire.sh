#!/bin/bash
# Setup script for Vampire theorem prover

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VAMPIRE_DIR="$ROOT_DIR/.vampire"

echo "Setting up Vampire..."

# Create directory
mkdir -p "$VAMPIRE_DIR"
cd "$VAMPIRE_DIR"

# Download Vampire if not already present
if [ ! -f "vampire" ]; then
    echo "Downloading Vampire 5.0.0..."
    wget -q https://github.com/vprover/vampire/releases/download/v5.0.0/vampire-Linux-X64.zip

    echo "Extracting..."
    python3 -m zipfile -e vampire-Linux-X64.zip .

    chmod +x vampire
    rm vampire-Linux-X64.zip

    echo "✓ Vampire installed"
else
    echo "✓ Vampire already installed"
fi

# Test Vampire
echo "Testing Vampire..."
./vampire --version | head -1

echo "✓ Vampire setup complete"
