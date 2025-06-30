#!/bin/bash
# Script to remove superfluous files from the Rust implementation

echo "=== Cleaning up superfluous files ==="

# Remove duplicate zero-copy implementations
echo "Removing duplicate zero-copy implementations..."
rm -f src/bindings/python/zero_copy_array.rs
rm -f src/bindings/python/zero_copy_bindings.rs

# Remove debug files
echo "Removing debug files..."
rm -f src/rules/superposition_debug.rs

# Remove test file from wrong location
echo "Removing misplaced test file..."
rm -f src/test_basic_parser.rs

# Also remove the import from lib.rs
echo "Updating lib.rs to remove test import..."
sed -i '/mod test_basic_parser;/d' src/lib.rs

# Remove test file that should be in tests/
echo "Removing test file from saturation module..."
rm -f src/saturation/test_superposition.rs

echo ""
echo "=== Files removed ==="
echo "- src/bindings/python/zero_copy_array.rs (redundant zero-copy attempt)"
echo "- src/bindings/python/zero_copy_bindings.rs (redundant zero-copy attempt)"
echo "- src/rules/superposition_debug.rs (debug code)"
echo "- src/test_basic_parser.rs (misplaced test)"
echo "- src/saturation/test_superposition.rs (test in wrong location)"

echo ""
echo "The main zero-copy implementation is in src/bindings/python/array_bindings.rs"
echo "using Box<[T]> for stable memory addresses."