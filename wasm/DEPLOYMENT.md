# Deployment Guide

## Local Development

1. Install dependencies:
   ```bash
   # Install Rust if needed
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install wasm-pack
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

2. Build the WASM module:
   ```bash
   cd wasm
   ./build.sh
   ```

3. Test locally:
   ```bash
   cd dist
   python3 -m http.server 8000
   # Visit http://localhost:8000
   ```

## GitHub Pages Deployment

1. Enable GitHub Pages in your repository settings:
   - Go to Settings → Pages
   - Source: Deploy from GitHub Actions

2. The workflow will automatically deploy when you:
   - Push changes to `main` branch
   - Modify files in `wasm/` or `rust/`

3. Your site will be available at:
   ```
   https://[your-username].github.io/proofatlas/
   ```

## Manual Deployment

If you want to deploy to a different hosting service:

1. Build the project:
   ```bash
   cd wasm
   ./build.sh
   ```

2. Upload the contents of `wasm/dist/` to your web server

3. Ensure your server has proper MIME types:
   - `.wasm` → `application/wasm`
   - `.js` → `application/javascript`

## Customization

- Edit `index.html` for UI changes
- Modify `style.css` for styling
- Update examples in `app.js`
- Change prover options in `src/lib.rs`

## Troubleshooting

- **CORS errors**: WASM modules must be served over HTTP(S), not file://
- **Module not loading**: Check browser console for errors
- **Performance issues**: Build with `--release` flag for optimization