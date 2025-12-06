# ProofAtlas Web Service

A client-side theorem prover that runs entirely in the browser using WebAssembly.

## Architecture

1. **Rust Core** → Compiled to WASM
   - TPTP parser
   - Saturation-based prover
   - Proof generation

2. **Web Interface**
   - Input editor with TPTP syntax highlighting
   - Real-time proving with progress updates
   - Proof visualization
   - Example problems

3. **Deployment**
   - Static site hosted on GitHub Pages
   - No server required - all computation in browser
   - Easy to fork and customize

## Building

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM module (from wasm directory)
cd wasm
wasm-pack build --target web

# Serve locally for testing
python3 -m http.server 8000
# Then visit http://localhost:8000/
```

## Usage

Visit the hosted site and:
1. Enter or select a TPTP problem
2. Click "Prove"
3. View the proof or saturation result

The prover runs entirely in your browser with no data sent to any server.