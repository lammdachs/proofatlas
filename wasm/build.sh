#!/bin/bash

# Build WASM module
echo "Building WASM module..."
cargo build --target wasm32-unknown-unknown --release

# Generate JS bindings
echo "Generating JS bindings..."
~/.cargo/bin/wasm-bindgen target/wasm32-unknown-unknown/release/proofatlas_wasm.wasm \
    --out-dir pkg \
    --target web \
    --no-typescript

# Copy files for deployment
echo "Preparing deployment files..."
mkdir -p dist
cp index.html style.css app.js dist/
cp -r pkg dist/

# Copy examples
echo "Copying examples..."
cp -r examples dist/

echo "Build complete! Files are in the dist/ directory"
echo "To test locally, run: cd dist && python3 -m http.server 8000"