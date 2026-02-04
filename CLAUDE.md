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
│   │       ├── fol/            # FOL types: terms, literals, clauses, substitutions, KBO, interner
│   │       ├── inference/      # Inference rules, derivation tracking, proof types
│   │       ├── selection/      # Selection strategies, graph building, proof trace (tch-rs ML)
│   │       ├── saturation/     # Saturation loop, trace, forward/backward subsumption, profiling
│   │       ├── parser/         # TPTP parser with FOF→CNF conversion (with timeout)
│   │       ├── unification/    # Most General Unifier (MGU) computation
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
│   └── setup_*.py              # Setup TPTP, Vampire, SPASS
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
LIBTORCH_USE_PYTORCH=1 maturin develop    # Build and install into Python environment
LIBTORCH_USE_PYTORCH=1 cargo test         # Run Rust tests
```

**Note:** The `python` and `ml` Cargo features are enabled by default. The WASM crate opts out via `default-features = false`. Set `LIBTORCH_USE_PYTORCH=1` to let torch-sys find libtorch from your PyTorch installation.

### Running the Prover

```bash
proofatlas problem.p                          # Basic usage
proofatlas problem.p --config time_sel21      # With preset
proofatlas problem.p --timeout 30             # With timeout
proofatlas --list                             # List available presets
```

### Tests

```bash
# Set up environment for cargo tests
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib

cargo test                               # All Rust tests
cargo test fol                           # Specific module
cargo test --test '*'                    # Integration tests only
cargo test -- --nocapture                # With output

pytest python/tests/ -v                  # Python tests (no env vars needed)
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

### Symbol Interning
All symbol names (variables, constants, functions, predicates) are interned into an `Interner` that maps strings to compact integer IDs:
- `VariableId`, `ConstantId`, `FunctionId`, `PredicateId` - 4-byte `Copy` types
- `Interner` is created during parsing and passed through the prover
- Substitution uses `HashMap<VariableId, Term>` for O(1) lookups instead of string hashing
- Symbol comparison is integer comparison (O(1) vs O(n) string comparison)
- JSON serialization resolves IDs back to strings

The interner is problem-scoped (not global) for WASM compatibility and clean memory management.

### Unification
Uses eager substitution propagation. When adding a new binding, all existing substitutions are immediately updated to prevent unsound inferences.

### Variable Renaming
Variables from different clauses are renamed to avoid capture (e.g., X becomes X_c10 for clause 10). New variable names are interned during inference.

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

### Polymorphic Rule Architecture
The saturation loop uses a modular, polymorphic architecture where rules are registered and applied generically:

**SimplificationRule trait** (`saturation/rule.rs`):
- `simplify_forward()`: Simplify/delete clause in N using U∪P
- `simplify_backward()`: Simplify clauses in U∪P using new clause
- Implementations: `TautologyRule`, `DemodulationRule`, `SubsumptionRule`

**GeneratingInferenceRule trait** (`saturation/rule.rs`):
- `generate()`: Generate inferences with given clause and clauses in P
- Implementations: `ResolutionRule`, `SuperpositionRule`, `FactoringRule`, `EqualityResolutionRule`, `EqualityFactoringRule`

All rules return `Vec<ProofStateChange>` for atomic state modifications:
- `New { clause, derivation }`: Add new clause to N
- `DeleteN/U/P { clause_idx, rule_name }`: Delete from respective set (simplification)
- `Transfer { clause_idx }`: Move clause N→U (implicit N removal)
- `Select { clause_idx }`: Move clause U→P (implicit U removal)

This architecture enables adding new rules without modifying the main loop.

### Derivation Tracking
The `Derivation` struct tracks how each clause was derived:
```rust
pub struct Derivation {
    pub rule_name: String,
    pub premises: Vec<usize>,
}
```

Rules construct derivations directly: `Derivation { rule_name: "Resolution".into(), premises: vec![p1, p2] }`. Only `Derivation::input()` helper exists for input clauses (no premises).

### Event Log
The saturation state maintains an event log (`Vec<ProofStateChange>`) as the single source of truth for derivations. All clause additions (`New`) include the derivation info. This enables:
- **Proof extraction**: `extract_proof()` builds a derivation map from the event log and traces back from the empty clause
- **Training data extraction**: Replay events to reconstruct clause sets and label by proof membership
- **Selection context tracking**: `SelectionTrainingExample` captures which clauses were available at each selection

The `EventLogReplayer` utility reconstructs N/U/P sets at any point by replaying events.

### Profiling
`SaturationConfig::enable_profiling` (default `false`) enables structured profiling of the saturation loop. When enabled, `saturate()` returns `(SaturationResult, Option<SaturationProfile>)` with timing and counting data for every phase:

- **Phase timings**: forward simplification, clause selection, inference generation, inference addition
- **Sub-phase timings**: forward/backward demodulation, forward/backward subsumption
- **Per-rule counts and timings**: resolution, superposition, factoring, equality resolution, equality factoring, demodulation
- **Aggregate counters**: iterations, clauses generated/added/subsumed/demodulated, tautologies deleted, max set sizes
- **Selector stats**: name, cache hits/misses, embed/score time (populated from `ClauseSelector::stats()`)

Zero overhead when disabled: all instrumentation is gated on `Option::None`, costing a single predicted-not-taken branch per instrumentation point.

`SaturationProfile` implements `serde::Serialize` with `Duration` fields serialized as `f64` seconds. From Python, `run_saturation()` returns `(proof_found, status, profile_json, trace_json)`. Pass `enable_profiling=True` to populate the profile.

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
