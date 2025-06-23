#!/bin/bash
# Example usage of TPTP extraction scripts

# Ensure conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "proofatlas" ]]; then
    echo "Please activate the proofatlas conda environment first:"
    echo "conda activate proofatlas"
    exit 1
fi

# Load environment variables
source .env

echo "TPTP extraction examples using environment variables"
echo "TPTP_PATH: $TPTP_PATH"
echo "DATASETS_DIR: $DATASETS_DIR"
echo ""

# Example 1: Extract first 10 problems using simple version
echo "Example 1: Extract first 10 TPTP problems (simple version)"
echo "python scripts/extract_tptp_to_json_simple.py --max-files 10"
echo ""

# Example 2: Extract all PUZ domain problems using parallel version
echo "Example 2: Extract all PUZ domain problems (parallel version)"
echo "python scripts/extract_tptp_to_json.py --domain PUZ --workers 4"
echo ""

# Example 3: Extract with individual file saving
echo "Example 3: Extract with individual JSON files"
echo "python scripts/extract_tptp_to_json_simple.py --save-individual --max-files 100"
echo ""

# Example 4: Custom output directory
echo "Example 4: Extract to custom output directory"
echo "python scripts/extract_tptp_to_json.py --output-dir ./my_extracted_problems"
echo ""

# Example 5: Full extraction with defaults (uses env vars)
echo "Example 5: Full extraction using defaults from environment variables"
echo "python scripts/extract_tptp_to_json.py"
echo ""

echo "Note: Output will be saved to $DATASETS_DIR/tptp_json by default"