# CLAUDE.md

## Project Overview

ProofAtlas is a saturation-based theorem prover in Rust with Python bindings for ML-based clause selection. Rust core in `crates/proofatlas/`, WASM bindings in `crates/proofatlas-wasm/`, Python package in `python/proofatlas/`, SvelteKit web UI in `web/`, orchestration scripts in `scripts/`.

## Commands

```bash
# Building
LIBTORCH_USE_PYTORCH=1 maturin develop        # Build Rust + install Python package
LIBTORCH_USE_PYTORCH=1 cargo test              # Rust tests

# Running
proofatlas problem.p                           # Prove a problem
proofatlas problem.p --config time             # With preset (--list to see all)

# Tests (set env vars first)
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib
cargo test                                     # All Rust tests
pytest python/tests/ -v                        # Python tests (no env vars needed)

# Training & Benchmarking
proofatlas-train --config gcn_mlp              # Train model (--use-cuda for GPU)
proofatlas-bench --config gcn_mlp              # Evaluate (requires trained weights)

# Web interface
wasm-pack build --target web --out-dir ../../web/static/pkg crates/proofatlas-wasm
cd web && npm install && npm run build         # Build SvelteKit static site
proofatlas-web                                 # Serve on port 8000
cd web && npm run dev                          # Dev server (port 5173)
```

All web static assets (data files, WASM pkg) must be under `web/static/` for SvelteKit to include them in the build. Scripts write training/benchmark data to `web/static/data/`.

## Correctness Rules

- **Soundness**: The prover must NEVER find a proof for a satisfiable problem. If it does, there is a bug.
- Every proof is independently verified via `verify_proof()`.
- When proof search times out, do NOT conclude "the proof is difficult." Ask the user — it may be a bug, wrong strategy, or missing inference rule.

## Testing with TPTP Problems

- Problems in `.tptp/TPTP-v9.0.0/Problems/`, simple ones: PUZ001-1.p, SYN000-1.p, GRP001-1.p
- Check problem status in file header: **Unsatisfiable** = should find proof, **Satisfiable** = must NOT find proof
- Problem sets defined in `configs/tptp.json`: `bench` (default, 11k problems), `test` (PUZ only, 123), `all` (13k)

## File System

- Create temporary files in current directory, not /tmp
- Gitignored data dirs: `.data/` (traces, runs), `.tptp/`, `.weights/`, `.vampire/`, `.spass/`
