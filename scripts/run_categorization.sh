#!/bin/bash

# Script to categorize TPTP problems and prepare for benchmarking

echo "Categorizing TPTP problems..."
echo "This may take a few minutes depending on the size of your TPTP installation."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Check if TPTP directory exists
TPTP_DIR="$BASE_DIR/.tptp/TPTP-v9.0.0/Problems"
if [ ! -d "$TPTP_DIR" ]; then
    echo "Error: TPTP directory not found at $TPTP_DIR"
    echo "Please ensure TPTP is installed in .tptp/"
    exit 1
fi

# Run the categorization script
python3 "$SCRIPT_DIR/categorize_tptp_problems.py"

echo ""
echo "Categorization complete!"
echo "Problem lists have been created in: $BASE_DIR/.data/benchmark_lists/"
echo ""
echo "You can now run benchmarks with:"
echo "  python3 $SCRIPT_DIR/benchmark_against_vampire.py"
echo ""
echo "Example commands:"
echo "  # Run on unit equality problems with 5 second timeout"
echo "  python3 $SCRIPT_DIR/benchmark_against_vampire.py --categories unit_equalities --timeout 5"
echo ""
echo "  # Run on first 100 CNF problems without equality"
echo "  python3 $SCRIPT_DIR/benchmark_against_vampire.py --categories cnf_without_equality --max-problems 100"
echo ""
echo "  # Run on all categories with 10 second timeout"
echo "  python3 $SCRIPT_DIR/benchmark_against_vampire.py --timeout 10"