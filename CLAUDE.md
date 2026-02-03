# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

ProofAtlas is a high-performance theorem prover in Rust with Python bindings. It implements saturation-based proving with superposition calculus for equality reasoning, plus ML-based clause selection.

## Project Structure

```
proofatlas/
│
├── crates/
│   ├── proofatlas/             # Core theorem prover (Rust)
│   │   └── src/
│   │       ├── fol/            # Pure FOL types: terms, literals, clauses, substitutions, KBO
│   │       ├── inference/      # Inference rules, derivation tracking, proof types
│   │       ├── selection/      # Literal + clause selection strategies (tch-rs ML)
│   │       ├── saturation/     # Saturation loop, trace, forward/backward subsumption, profiling
│   │       ├── parser/         # TPTP parser with FOF→CNF conversion (with timeout)
│   │       ├── unification/    # Most General Unifier (MGU) computation
│   │       ├── ml/             # Graph building from clauses
│   │       └── json.rs         # JSON serialization types
│   │
│   └── proofatlas-wasm/        # WebAssembly bindings for browser execution
│
├── python/proofatlas/          # Python package
│   ├── cli/                    # Command-line interface (bench entry point)
│   ├── ml/                     # Training configs, data loading, training loops
│   └── selectors/              # PyTorch model implementations (GCN, Sentence)
│
├── web/                        # Web frontend (HTML/CSS/JS)
│
├── configs/                    # JSON configuration for provers, training, benchmarks
│
├── scripts/                    # Utility scripts
│   ├── setup.py                # One-command project setup
│   ├── bench.py                # Multi-prover benchmarking with trace collection
│   ├── export.py               # Export results for web display
│   └── setup_*.py              # Setup libtorch, TPTP, Vampire, SPASS
│
├── .data/                      # Runtime data (gitignored)
│   ├── traces/                 # Proof search traces
│   └── runs/                   # Benchmark results
├── .tptp/                      # TPTP problem library (gitignored)
├── .weights/                   # Trained model weights (gitignored)
├── .vampire/                   # Vampire prover binary (gitignored)
└── .spass/                     # SPASS prover binary (gitignored)
```

## Commands

### Building

```bash
maturin develop                     # Build and install into Python environment
```

**Note:** The `python` and `ml` Cargo features are enabled by default. The WASM crate opts out via `default-features = false`. After switching Python environments, re-run `python scripts/setup_libtorch.py` to update libtorch paths in `.cargo/config.toml`.

### Running the Prover

```bash
./target/release/prove <tptp_file> [options]

# Options:
#   --timeout <seconds>        Timeout (default: 300s)
#   --max-clauses <n>          Clause limit (default: 10000)
#   --literal-selection <n>    0=all, 20=maximal, 21=unique
```

### Tests

```bash
cargo test                               # All Rust tests
cargo test fol                           # Specific module
cargo test --test '*'                    # Integration tests only
cargo test -- --nocapture                # With output

python -m pytest python/tests/ -v        # Python tests
```

### Benchmarking

```bash
proofatlas-bench                              # Run all presets
proofatlas-bench --config time_sel21          # Run specific preset
proofatlas-bench --retrain                    # Retrain ML models
proofatlas-bench --status                     # Check job status
```

## Problem Sets

Problem sets are defined in `configs/tptp.json` and filter TPTP problems for benchmarking. The default is set in `defaults.problem_set`.

### Available Filters

| Filter | Type | Description |
|--------|------|-------------|
| `status` | list | Problem status: `["unsatisfiable"]`, `["satisfiable"]` |
| `format` | list | Input format: `["cnf"]`, `["fof"]`, `["cnf", "fof"]` |
| `domains` | list | Include only these domains: `["GRP", "PUZ"]` |
| `exclude_domains` | list | Exclude these domains: `["CSR", "HWV", "SWV"]` |
| `max_rating` | float | Maximum TPTP difficulty rating (0.0-1.0) |
| `max_clauses` | int | Maximum number of clauses in problem |
| `max_term_depth` | int | Maximum term nesting depth |
| `max_clause_size` | int | Maximum literals per clause |
| `has_equality` | bool | `true` = only equality, `false` = no equality |
| `is_unit_only` | bool | `true` = only unit clauses, `false` = has non-unit |

### Example Problem Set

```json
{
  "problem_sets": {
    "unit_equality": {
      "description": "Unit equality problems",
      "status": ["unsatisfiable"],
      "format": ["cnf"],
      "has_equality": true,
      "is_unit_only": true,
      "max_clauses": 500,
      "max_term_depth": 8,
      "domains": ["GRP", "RNG", "LAT"]
    }
  }
}
```

## Testing the Theorem Prover

When testing with TPTP problems:
- Problems are in `.tptp/TPTP-v9.0.0/Problems/`
- Start with simple problems: PUZ001-1.p, SYN000-1.p, GRP001-1.p
- Check problem status in file header:
  - **Unsatisfiable** = should find proof
  - **Satisfiable** = should saturate or timeout (must NOT find proof)
- Test satisfiable problem: RNG025-9.p (should NOT find proof)

## Soundness and Completeness

Critical correctness properties:
- **Soundness**: Never find a proof for a satisfiable problem
- **Completeness**: Find proofs for unsatisfiable problems (or exhaust resources)

If the prover finds a proof for a satisfiable problem, there is a bug.

## Key Implementation Details

### Unification
Uses eager substitution propagation. When adding a new binding, all existing substitutions are immediately updated to prevent unsound inferences.

### Variable Renaming
Variables from different clauses are renamed to avoid capture (e.g., X becomes X_c10 for clause 10).

### Superposition Calculus
- Only applies to positive equalities
- Respects KBO term ordering
- Collects literals from correct parent clauses

### Literal Selection
Four strategies (Hoder et al. "Selecting the selection" 2016):
- **0**: Select all literals (default)
- **20**: Select all maximal literals
- **21**: Unique maximal, else neg max-weight, else all maximal
- **22**: Neg max-weight, else all maximal

### Demodulation
- Uses one-way matching (only rewrite rule variables can be substituted)
- Enforces ordering constraint lσ ≻ rσ
- Forward demodulation: simplify new clauses before adding
- Backward demodulation: when unit equality selected, simplify existing clauses

### CNF Conversion
- FOF formulas converted via distribution
- Has timeout parameter to prevent exponential blowup on deeply nested formulas
- Timeout returns error, not silent failure

### Subsumption
Tiered approach: duplicates → variants → units → small clauses → greedy

### Profiling
`SaturationConfig::enable_profiling` (default `false`) enables structured profiling of the saturation loop. When enabled, `saturate()` returns `(SaturationResult, Option<SaturationProfile>)` with timing and counting data for every phase:

- **Phase timings**: forward simplification, clause selection, inference generation, inference addition
- **Sub-phase timings**: forward/backward demodulation, forward/backward subsumption
- **Per-rule counts and timings**: resolution, superposition, factoring, equality resolution, equality factoring, demodulation
- **Aggregate counters**: iterations, clauses generated/added/subsumed/demodulated, tautologies deleted, max set sizes
- **Selector stats**: name, cache hits/misses, embed/score time (populated from `ClauseSelector::stats()`)

Zero overhead when disabled: all instrumentation is gated on `Option::None`, costing a single predicted-not-taken branch per instrumentation point.

`SaturationProfile` implements `serde::Serialize` with `Duration` fields serialized as `f64` seconds. From Python, pass `enable_profiling=True` to `run_saturation()` to receive the profile as a JSON string in the third return element: `(proof_found, status, profile_json)`.

## ML Architecture

**Workflow**: Train in PyTorch → Export to TorchScript → Load in Rust/tch-rs

| Selector | Rust | PyTorch | Notes |
|----------|------|---------|-------|
| age_weight | ✓ | - | Heuristic, no training |
| gcn | ✓ | ✓ | Graph Convolutional Network |
| sentence | ✓ | ✓ | Sentence transformer (MiniLM) |

Selectors implement the `ClauseSelector` trait. The optional `stats()` method returns `SelectorStats` (cache hits/misses, embed/score time). `CachingSelector` tracks these automatically; `AgeWeightSelector` returns `None`.

ML selectors use tch-rs (PyTorch C++ bindings) for GPU-accelerated inference and are enabled by default. Models are exported as TorchScript (`.pt` files). At runtime, libtorch (CPU and CUDA if available) is preloaded from the user's PyTorch installation via `python/proofatlas/__init__.py`.

## Analysis Guidelines

When proof search times out or takes many steps, do NOT conclude "the proof is difficult." Ask the user for analysis. Possible causes:
- Bug in implementation
- Wrong selection strategy
- Missing inference rules
- Incorrect problem formulation

## File System

- Create temporary files in current directory, not /tmp
- Clean up temporary files when done
- TPTP problems: `.tptp/TPTP-v9.0.0/Problems/`
