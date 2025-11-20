#!/bin/bash
# Setup script for PyRes theorem prover

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYRES_DIR="$ROOT_DIR/.pyres"

echo "Setting up PyRes..."

# Create directory
mkdir -p "$PYRES_DIR"
cd "$PYRES_DIR"

# Clone PyRes if not already present
if [ ! -d ".git" ]; then
    echo "Cloning PyRes from GitHub..."
    git clone https://github.com/eprover/PyRes.git .
    echo "✓ PyRes cloned"
else
    echo "✓ PyRes already installed"
fi

# Make scripts executable
chmod +x pyres-cnf.py pyres-fof.py 2>/dev/null || true

# Test PyRes
echo "Testing PyRes..."
python3 pyres-cnf.py --version 2>&1 | head -3 || echo "PyRes ready (no --version flag)"

echo "✓ PyRes setup complete"
