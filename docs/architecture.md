# Architecture Overview

ProofAtlas is a high-performance theorem prover with ML-guided clause selection. This document describes the system architecture and data flow.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         ProofAtlas                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Parser     │───▶│  Saturation  │───▶│  Proof/Result    │   │
│  │   (TPTP)     │    │    Loop      │    │                  │   │
│  └──────────────┘    └──────┬───────┘    └──────────────────┘   │
│                             │                                    │
│                    ┌────────┴────────┐                          │
│                    ▼                 ▼                          │
│             ┌────────────┐    ┌────────────┐                    │
│             │  Clause    │    │  Inference │                    │
│             │  Selector  │    │   Rules    │                    │
│             └─────┬──────┘    └────────────┘                    │
│                   │                                              │
│          ┌────────┴────────┐                                    │
│          ▼                 ▼                                    │
│   ┌────────────┐    ┌────────────┐                              │
│   │ Age-Weight │    │  ML Model  │                              │
│   │ Heuristic  │    │  (tch-rs)  │                              │
│   └────────────┘    └────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Python Bindings (PyO3)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Python Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   CLI Tools  │    │  ML Training │    │  Benchmarking    │   │
│  │              │    │  (PyTorch)   │    │                  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Rust Crate Structure

```
crates/proofatlas/src/
├── lib.rs                 # Public API exports
├── bin/                   # CLI binaries
│   └── prove.rs           # Main prover binary
│
├── core/                  # Core data structures
│   ├── problem.rs         # ArrayProblem (CSR representation)
│   ├── symbol_table.rs    # String interning
│   ├── builder.rs         # Problem construction
│   ├── proof.rs           # Proof tracking
│   └── parser_convert.rs  # Parse-to-array conversion
│
├── parser/                # TPTP parsing
│   ├── tptp.rs            # High-level TPTP API
│   ├── tptp_parser.rs     # Parser implementation
│   ├── parse_types.rs     # Intermediate representations
│   ├── fof.rs             # FOF formula handling
│   └── prescan.rs         # Fast file scanning
│
├── rules/                 # Inference rules
│   ├── common.rs          # Shared types (InferenceResult)
│   ├── resolution.rs      # Binary resolution
│   ├── factoring.rs       # Factoring
│   ├── superposition.rs   # Superposition (equality)
│   ├── equality_resolution.rs
│   ├── equality_factoring.rs
│   └── demodulation.rs    # Simplification
│
├── saturation/            # Proof search
│   ├── loop.rs            # Given-clause algorithm
│   ├── literal_selection.rs
│   ├── clause_selection.rs
│   ├── subsumption.rs     # Redundancy elimination
│   └── unification.rs     # MGU computation
│
├── selectors/             # Clause selection strategies
│   ├── age_weight.rs      # Age-weight heuristic
│   ├── gcn.rs             # GCN model (tch-rs)
│   ├── sentence.rs        # Sentence transformer (tch-rs)
│   └── cached.rs          # Embedding cache
│
├── ml/                    # ML infrastructure
│   └── graph.rs           # Clause-to-graph conversion
│
└── python_bindings.rs     # PyO3 bindings
```

## Data Flow

### 1. Parsing

```
TPTP File → Parser → ParsedFormula → ArrayProblem
```

- TPTP files are parsed into an intermediate representation
- FOF formulas are converted to CNF (with timeout protection)
- Clauses are converted to CSR (Compressed Sparse Row) format
- Symbols are interned in a `SymbolTable` for O(1) comparison

### 2. Saturation Loop

```
ArrayProblem → Saturation Loop → Proof | Saturated | ResourceLimit
```

The given-clause algorithm:
1. **Select**: Choose a clause from the unprocessed set
2. **Infer**: Generate new clauses using inference rules
3. **Simplify**: Apply forward/backward subsumption, demodulation
4. **Add**: Add non-redundant clauses to unprocessed set
5. **Repeat**: Until empty clause found or resources exhausted

### 3. Clause Selection

```
Unprocessed Clauses → Selector → Selected Clause
```

**Heuristic (Age-Weight)**:
- With probability `p`: select oldest clause (FIFO)
- With probability `1-p`: select lightest clause (by symbol count)

**ML-Based (GCN/MLP)**:
1. Convert clauses to graphs
2. Run GNN to produce embeddings
3. Score embeddings with MLP
4. Sample from softmax distribution

### 4. ML Training Pipeline

```
Proof Traces → PyTorch Training → TorchScript → tch-rs Inference (GPU)
```

1. **Trace Collection**: During proof search, record clause graphs and labels
2. **Training**: Train GNN in PyTorch using contrastive loss
3. **Export**: Save model to TorchScript format (.pt)
4. **Inference**: Load model in Rust/tch-rs for GPU-accelerated scoring

## Key Design Decisions

### CSR Representation

Terms and clauses use Compressed Sparse Row format:
- Excellent cache locality
- Minimal memory overhead
- Fast child iteration
- No pointer chasing

### String Interning

All symbols (predicates, functions, constants) are interned:
- O(1) equality comparison
- Reduced memory for repeated symbols
- Cache-friendly symbol access

### Modular Inference Rules

Each rule is a separate module with common interface:
- Easy to add new rules
- Independent testing
- Clear separation of concerns

### Dual Runtime (Rust/Python)

- **Rust**: High-performance inference, production use
- **Python**: Training, experimentation, visualization

## Module Dependencies

```
                    ┌─────────┐
                    │  lib.rs │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────┐     ┌────────────┐   ┌────────┐
    │ parser │     │ saturation │   │ python │
    └────┬───┘     └─────┬──────┘   └────────┘
         │               │
         │        ┌──────┴──────┐
         │        │             │
         ▼        ▼             ▼
    ┌────────┐  ┌───────┐  ┌───────────┐
    │  core  │  │ rules │  │ selectors │
    └────────┘  └───────┘  └─────┬─────┘
                                 │
                                 ▼
                            ┌────────┐
                            │   ml   │
                            └────────┘
```

**Dependency rules**:
- `core`: No dependencies on other modules
- `rules`: Depends on `core`
- `saturation`: Depends on `core`, `rules`
- `selectors`: Depends on `core`, `ml`
- `parser`: Depends on `core`
- `python`: Depends on all modules

## Performance Considerations

### Memory
- CSR format minimizes allocations
- String interning reduces duplication
- Pre-allocated buffers in builder

### CPU
- No pointer chasing in term traversal
- Cache-friendly data layout
- Lazy substitution application where possible

### Parallelism (Future)
- Subsumption checking is parallelizable
- Independent inferences can run concurrently
- Graph construction for ML is batch-friendly

## Python Integration

### PyO3 Bindings

The `python_bindings.rs` module exposes:
- `ProofAtlas`: Main proof state management
- `ClauseInfo`: Clause inspection
- Graph export for ML training

### Workflow

```python
from proofatlas import ProofAtlas

state = ProofAtlas()
state.add_clauses_from_tptp(content, tptp_root, timeout)
state.set_literal_selection("21")

proof_found, status = state.prove(
    max_clauses=10000,
    timeout=60.0,
    age_weight_ratio=0.167,
    encoder=None,
    scorer=None,
    weights_path=None,
)

if proof_found:
    examples = state.extract_training_examples()
    graphs = state.clauses_to_graphs([e.clause_idx for e in examples])
```

## File Locations

| Component | Location |
|-----------|----------|
| Rust prover | `crates/proofatlas/src/` |
| Python package | `python/proofatlas/` |
| CLI tools | `python/proofatlas/cli/` |
| PyTorch models | `python/proofatlas/selectors/` |
| Training | `python/proofatlas/ml/` |
| Configurations | `configs/` |
| Documentation | `docs/` |
| WASM bindings | `crates/proofatlas-wasm/` |
| Web frontend | `web/` |
