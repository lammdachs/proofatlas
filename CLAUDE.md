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
│   │       ├── logic/          # FOL types and manipulation
│   │       │   ├── core/       # term.rs, literal.rs, clause.rs, position.rs
│   │       │   ├── ordering/   # kbo.rs, orient_equalities.rs
│   │       │   ├── unification/# mgu.rs, matching.rs, substitution.rs, scoped.rs
│   │       │   ├── interner.rs # Symbol interning
│   │       │   ├── literal_selection.rs  # LiteralSelector trait + impls
│   │       │   ├── clause_manager.rs     # ClauseManager: interner + selector + KBO
│   │       │   └── time_compat.rs        # WASM-compatible Instant
│   │       ├── simplifying/    # SimplifyingInference impls (tautology, subsumption, demodulation)
│   │       ├── generating/     # GeneratingInference impls (resolution, superposition, factoring, etc.)
│   │       ├── index/          # Index trait, IndexRegistry, SubsumptionChecker, SelectedLiteralIndex
│   │       ├── prover/         # Saturation engine
│   │       │   ├── mod.rs      # Prover struct with prove()/init()/step()/saturate()
│   │       │   └── profile.rs  # SaturationProfile
│   │       ├── selection/      # Clause selection strategies
│   │       │   ├── clause.rs   # ProverSink trait
│   │       │   ├── age_weight.rs # AgeWeightSink (age-weight heuristic)
│   │       │   ├── cached.rs   # ClauseEmbedder, EmbeddingScorer traits
│   │       │   ├── ml/         # ML model implementations (gcn, sentence, graph, features)
│   │       │   ├── pipeline/   # Backend compute service, ChannelSink, EmbedScoreModel
│   │       │   └── training/   # npz.rs (NpzWriter for trace output)
│   │       ├── parser/         # TPTP parser with FOF→CNF conversion (with timeout)
│   │       ├── atlas.rs        # ProofAtlas orchestrator (reusable across problems)
│   │       ├── config.rs       # ProverConfig, LiteralSelectionStrategy
│   │       └── state.rs        # SaturationState, StateChange, EventLog, traits
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
│   ├── bench.py                # Benchmark orchestration, CLI/daemon
│   ├── bench_jobs.py           # Job/daemon management, PID tracking, status display
│   ├── bench_provers.py        # Prover execution with ProofAtlasPool (persistent workers)
│   ├── run_all.py              # Full experiment orchestration (traces → train → eval → push)
│   ├── run_test.py             # Quick smoke test (test problem set, 1 epoch, all configs)
│   ├── train.py                # Standalone ML model training (extracted from bench.py)
│   ├── export.py               # Export results for web display
│   └── setup_*.py              # Setup TPTP, Vampire, SPASS, MiniLM base model
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
proofatlas problem.p --config time             # With preset
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

### Training & Benchmarking

```bash
proofatlas-train --config gcn_mlp                    # Train ML model
proofatlas-train --config gcn_mlp --use-cuda         # Train on GPU
proofatlas-train --config gcn_mlp --max-epochs 4     # Short test run

proofatlas-bench                              # Run all presets
proofatlas-bench --config gcn_mlp              # Run specific preset (requires trained weights)
proofatlas-bench --status                     # Check job status
```

## Problem Sets

Problem sets are defined in `configs/tptp.json` and filter TPTP problems for benchmarking. The default is set in `defaults.problem_set`. All sets share a base filter of CNF/FOF format with a 64kB total size limit (problem file + axiom files).

| Set | Description |
|-----|-------------|
| `all` | All CNF/FOF problems under 64kB (13,178 problems) |
| `bench` | Unsatisfiable and unknown problems (default, 11,094 problems) |
| `test` | PUZ domain only (123 problems) |

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
| `max_total_size_bytes` | int | Max combined size of problem + included axiom files |

### TPTP GitHub Mirror

A CNF/FOF subset of TPTP v9.0.0 is mirrored at `lammdachs/proofatlas-tptp-subset` for the web UI. The web frontend fetches problems from `raw.githubusercontent.com` when no local server is available. Problem metadata (`file_size`, `includes`, `total_size`) is extracted by `setup_tptp.py`.

## Testing the Theorem Prover

Five-layer testing strategy:

1. **Proof Verification**: Every proof is independently verified — `verify_proof()` on `Prover` checks each inference step. Each rule implements `verify()` on its trait (`GeneratingInference`/`SimplifyingInference`). `VerificationError` enum in `state.rs`.
2. **Calculus Compliance**: 41 tests in `test_calculus_compliance.rs` covering side conditions for all 8 inference rules.
3. **Property-Based Testing**: `proptest` tests in `logic/unification/proptest_tests.rs` and `logic/ordering/proptest_tests.rs` — unification soundness/symmetry, KBO transitivity/totality, matching correctness.
4. **TPTP Regression Suite**: 34 tests in `test_tptp_regression.rs` with `tptp_proof!`/`tptp_no_proof!` macros. Problem files in `tests/problems/`. Every proof runs `verify_proof()` automatically.
5. **WASM Tests**: 22 `wasm-bindgen-test` tests in `crates/proofatlas-wasm/tests/wasm_tests.rs`. Run with `wasm-pack test --headless --chrome crates/proofatlas-wasm`.

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

### Scoped Variables (Binary Inference)
Binary inference rules (resolution, superposition) use scoped variables instead of variable renaming. Each clause's variables are tagged with a scope (u8: 0 for first clause, 1 for second). `ScopedVar{scope, id}` distinguishes same-named variables from different clauses without cloning or interning. After unification succeeds, `flatten()` converts back to concrete Terms (scope 0 keeps original IDs, others get fresh `{name}_{scope}` names). Unary inferences (factoring, equality resolution/factoring) continue using regular `Substitution`.

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
- Forward demodulation: uses `DiscriminationTree` index for O(|term|) candidate filtering instead of scanning all unit equalities
- Backward demodulation: when unit equality selected, simplify existing clauses (still linear scan of U∪P)

### CNF Conversion
- FOF formulas converted via distribution
- Has timeout parameter to prevent exponential blowup on deeply nested formulas
- Timeout returns error, not silent failure

### Subsumption
Tiered approach: duplicates → variants → units → small clauses → greedy.

Candidate filtering uses a literal discrimination tree (`index/disc_tree.rs` shared core, `index/subsumption.rs` wrapper). Each literal is indexed under its (predicate, polarity) pair. Forward subsumption uses hit-count filtering with interleaved early exit and generation-counter dedup. Backward subsumption uses sorted Vec intersection across per-literal instance retrieval results. `MatchSubst` in `simplifying/subsumption.rs` is a flat-array matching substitution (`Vec<Option<&Term>>` indexed by VariableId) for zero-clone matching.

### Architecture

The prover is organized around a central `ProofAtlas` struct (`prover/mod.rs`) that orchestrates saturation:
- `ProofAtlas::new()` initializes all components (does not add clauses to state)
- `ProofAtlas::init()` adds initial clauses to N via `apply_change` (detects empty clause in input)
- `ProofAtlas::prove(&mut self) -> ProofResult` calls `init()` then runs the saturation loop to completion. The prover is retained after proving — all state (clauses, event log, profile, interner) is accessible via accessor methods.
- `ProofAtlas::step()` executes a single iteration (useful for debugging/visualization)
- Accessor methods: `interner()`, `event_log()`, `clauses()`, `profile()`, `extract_proof(idx)`
- `ProofResult` is a simple enum: `Proof { empty_clause_idx }`, `Saturated`, `ResourceLimit` — no cloned data, all state stays on the prover

**ClauseManager** (`logic/clause_manager.rs`): Centralizes clause-level operations:
- Symbol interning (`Interner`)
- Literal selection (`LiteralSelector` trait)
- Term ordering (`KBO`)
- Resource-limit state: `cancel` (shared `AtomicBool`), `start_time`, `timeout`, `memory_limit`, `baseline_rss_mb` — generating rules check these mid-inference via `stopped()` closures to break out of expensive candidate loops early
- Methods: `orient_equalities()`

**SaturationState** (`state.rs`): Lean data container holding:
- `clauses: Vec<Clause>` (append-only storage)
- `new`, `unprocessed`, `processed` clause sets (N/U/P)
- `event_log: Vec<StateChange>` (all modifications)

### Polymorphic Rule Architecture

Rules are **stateless** — they receive the full context at call time and do not maintain internal state or lifecycle hooks. The `IndexRegistry` handles all clause lifecycle events via methods that mirror `StateChange` variants: `on_add`, `on_transfer`, `on_delete`, `on_activate`.

**SimplifyingInference trait** (`state.rs`):
- `simplify_forward(clause_idx, &SaturationState, &ClauseManager, &mut IndexRegistry) -> Option<StateChange>`: Simplify/delete clause in N using U∪P
- `simplify_backward(clause_idx, &SaturationState, &ClauseManager, &mut IndexRegistry)`: Simplify clauses in U∪P using new clause
- Implementations in `simplifying/`: `TautologyRule`, `DemodulationRule`, `SubsumptionRule`

**GeneratingInference trait** (`state.rs`):
- `generate(given_idx, &SaturationState, &mut ClauseManager, &IndexRegistry)`: Generate inferences with given clause and clauses in P
- Implementations in `generating/`: `ResolutionRule`, `SuperpositionRule`, `FactoringRule`, `EqualityResolutionRule`, `EqualityFactoringRule`

**IndexRegistry** (`index/mod.rs`): Central registry owning all indices, routes clause lifecycle events:
- `SubsumptionChecker` (`index/subsumption.rs`): Literal discrimination tree + clause keys + unit tracking for subsumption
- `UnitEqualitiesIndex`: Tracks unit positive equalities for backward demodulation
- `DiscriminationTree` (`index/discrimination_tree.rs`): Trie index on rewrite rule LHS terms for O(|term|) forward demodulation candidate filtering (replaces linear scan of all unit equalities)
- `SelectedLiteralIndex` (`index/selected_literals.rs`): Maps (PredicateId, polarity) to processed clause entries for generating inference candidate filtering

All rules return `StateChange` values for atomic state modifications:
- `Add(Arc<Clause>, rule_name, premises)`: Add new clause to N
- `Simplify(clause_idx, Option<Arc<Clause>>, rule_name, premises)`: Remove clause and optionally replace it
- `Transfer(clause_idx)`: Move clause N→U
- `Activate(clause_idx)`: Move clause U→P

This architecture enables adding new rules without modifying the main loop.

### Derivation Tracking
Derivation information (rule name and premises) is stored directly in `StateChange::Add` and `StateChange::Simplify` tuple variants—there is no separate `Derivation` struct. The standalone `clause_indices(premises: &[Position]) -> Vec<usize>` helper extracts clause indices from premise positions for proof extraction.

### Event Log
The saturation state maintains an event log (`Vec<StateChange>`) as the single source of truth for derivations. All clause additions (`Add(clause, rule_name, premises)`) include the derivation info inline. This enables:
- **Proof extraction**: `extract_proof()` builds a derivation map from the event log and traces back from the empty clause
- **Training data extraction**: Per-clause lifecycle arrays (transfer_step, activate_step, simplify_step) written to NPZ by `save_trace()` in Rust; training samples random step k and reconstructs U_k/P_k from lifecycle arrays

### Literal Representation
Literals have a flat structure with `predicate`, `args`, and `polarity` directly on the `Literal` struct (no intermediate `Atom` wrapper). `Atom` still exists for FOF formula representation in the parser but is not used in clause-level operations.

### Position
`Position` (`logic/core/position.rs`) identifies a location within the clause store: a clause index plus a path into that clause's structure (literal index, then term path). Used in `StateChange` premises and throughout the system to reference inference sites.

### Profiling
`ProverConfig::enable_profiling` (default `false`) enables structured profiling of the saturation loop. When enabled, `prover.profile()` returns `Some(&SaturationProfile)` after `prove()` with timing and counting data for every phase:

- **Phase timings**: forward simplification, clause selection, inference generation, inference addition
- **Sub-phase timings**: forward/backward demodulation, forward/backward subsumption
- **Per-rule counts and timings**: resolution, superposition, factoring, equality resolution, equality factoring, demodulation
- **Aggregate counters**: iterations, clauses generated/added/subsumed/demodulated, tautologies deleted, max set sizes
- **Selector stats**: name, cache hits/misses, embed/score time (populated from `ProverSink::stats()`)

Zero overhead when disabled: all instrumentation is gated on `Option::None`, costing a single predicted-not-taken branch per instrumentation point.

`SaturationProfile` implements `serde::Serialize` with `Duration` fields serialized as `f64` seconds. From Python, `run_saturation()` returns `(proof_found, status, profile_json, trace_json)`. Pass `enable_profiling=True` to populate the profile.

### Clause Selection

**`ProverSink`** (`selection/clause.rs`): Signal-based interface. The prover pushes lifecycle events (`on_transfer`, `on_activate`, `on_simplify`) and requests selection via `select()`. Implementations track their own internal state from signals.
- `AgeWeightSink`: Heuristic age-weight ratio with internal `IndexMap<usize, usize>`
- `ChannelSink`: Sends signals via `mpsc` channel to a data processing thread (pipelined ML inference)

### Pipelined ML Inference

The primary ML inference path uses an in-process pipeline:

```
Prover --> ChannelSink --(mpsc)--> Data Processing Thread --> BackendHandle --> Backend
                                   (embedding cache,           (GPU/CPU models)
                                    softmax sampling)
```

- **`Backend`** (`selection/backend.rs`): Model-agnostic compute service with **lazy model loading**. Receives `ModelSpec` (model_id + factory closure) and loads models on first request using the requester's `use_cuda` device hint. Worker thread with 16 MiB stack. `Backend::new(specs)` for lazy loading, `Backend::from_models(models)` for backward compat.
- **`ModelSpec`**: Lazy model specification — `model_id` + `factory: Box<dyn FnOnce(bool) -> Result<Box<dyn Model>, String>>`. Factory called with `use_cuda` from first request for that model_id.
- **`BackendHandle`**: Cheaply cloneable, wraps `mpsc::Sender<BackendRequest>`. `submit_sync(id, model_id, data, use_cuda)` for blocking request-response with per-request device hint.
- **`DataProcessor`** implementations (`selection/pipeline/processors.rs`): 9 processors for all (encoder, scorer) combinations. Each specifies `embed_cuda`/`score_cuda` per request. GCN/features encoders request CPU; sentence encoders follow `use_cuda`.
- **`EmbedScoreModel`** (`selection/pipeline.rs`): Backend `Model` wrapping `ClauseEmbedder + EmbeddingScorer`. Receives `Arc<Clause>`, returns `f32` scores.
- **`ChannelSink`** (`selection/pipeline.rs`): `ProverSink` impl. On `select()`, sends `Select` signal and blocks for response. Owns data processing thread (joined on drop).
- **Data processing thread**: Receives `ProverSignal`s. On `Transfer`: submits clause to Backend, caches score. On `Select`: softmax-samples from cached scores.
- **Factory**: `create_ml_pipeline(embedder, scorer, temperature) -> ChannelSink`

## ML Architecture

**Workflow**: Train in PyTorch --> Export to TorchScript --> Load in Backend (CPU or GPU via `use_cuda`)

| Selector | Rust | PyTorch | Notes |
|----------|------|---------|-------|
| age_weight | Y | - | Heuristic, no training |
| gcn | Y | Y | Graph Convolutional Network |
| gcn_struct | Y | Y | GCN with features-only node info (no symbol embeddings) |
| features | Y | Y | 9D clause feature MLP |
| sentence | Y | Y | Sentence transformer (MiniLM) |

**Scorer types** (all support `forward(u_emb, p_emb=None)`):
- `mlp`: Simple feed-forward scorer
- `attention`: Multi-head self/cross-attention with learnable sentinel
- `transformer`: Full transformer block with cross-attention
- `cross_attention`: Dot-product cross-attention (ignores p_emb)

**Training and evaluation are separate steps:**
```bash
python scripts/train.py --config gcn_mlp       # Step 1: train model
proofatlas-bench --config gcn_mlp               # Step 2: evaluate (requires weights)
```

ML selectors use tch-rs (PyTorch C++ bindings) for inference and are enabled by default. Models are exported as TorchScript (`.pt` files). At runtime, libtorch is preloaded from the user's PyTorch installation via `python/proofatlas/__init__.py`.

### Trace Format (Per-Problem NPZ with Lifecycle Encoding)

Traces are stored as one NPZ file per problem (`.graph.npz` or `.sentence.npz`) in `.data/traces/{preset}/`. Written by `Prover.save_trace()` in Rust via `NpzWriter`. Per-clause lifecycle arrays enable step sampling at training time:

| Array | Type | Description |
|-------|------|-------------|
| `transfer_step[C]` | i32 | Step when clause entered U (-1 if never) |
| `activate_step[C]` | i32 | Step when clause activated U→P (-1 if never) |
| `simplify_step[C]` | i32 | Step when clause simplified/deleted (-1 if never) |
| `labels[C]` | u8 | 1 if clause is in the proof, 0 otherwise |
| `num_steps[1]` | i32 | Total activation steps |
| `clause_features[C,9]` | f32 | 9 clause features |

Graph traces additionally include `node_features`, `edge_src/dst`, `node_offsets/edge_offsets`, and optionally `node_embeddings` (pre-computed MiniLM, 384-D) + `node_sentinel_type`. Sentence traces include `clause_embeddings` (pre-computed MiniLM, 384-D).

**Reconstruction at step k**: `U_k = {i : transferred ≤ k AND not yet activated at k AND not simplified}`, `P_k = {i : activated < k AND not simplified}`.

**`MiniLMEncoderModel`** (`selection/ml/sentence.rs`): Rust Backend Model wrapping base MiniLM (TorchScript) + tokenizer for pre-computing 384-D embeddings at trace time.

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
