# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProofAtlas is a high-performance theorem prover implemented in Rust with Python bindings. The core implementation is in Rust, providing a complete saturation-based theorem prover with superposition calculus for equality reasoning.

## Testing the Theorem Prover

When asked to test the theorem prover:
- Use actual TPTP problems from `.tptp/TPTP-v9.0.0/Problems/`
- Start with simpler problems (e.g., PUZ, SYN with small numbers)
- Check problem status in the file header (Unsatisfiable = should find proof, Satisfiable = should saturate/timeout)
- Test problems from different domains to ensure broad coverage
- Run with: `cargo run --bin prove -- <problem_file> [options]`
- For release build: `cargo build --release && ./target/release/prove <problem_file> [options]`

### Common Test Problems
- Simple puzzles: PUZ001-1.p, PUZ002-1.p
- Propositional: SYN000-1.p, SYN074-1.p
- Groups: GRP001-1.p (simple), GRP001-4.p (harder)
- Rings: RNG025-9.p (satisfiable - should NOT find proof)

## Important: Soundness and Completeness

When working with theorem provers, remember:
- **Soundness**: The prover should NEVER find a proof for a satisfiable problem
- **Completeness**: If a problem is unsatisfiable (a theorem) and the prover saturates without finding a proof, this indicates incompleteness
- A complete theorem prover will either find a proof for unsatisfiable problems or run out of resources (time/memory)
- Recent bug fix: Unification now uses eager substitution propagation to prevent unsound inferences
- Subsumption: Uses a pragmatic tiered approach (duplicates → variants → units → small clauses → greedy) for good performance

## Environment Setup and Commands

### Building and Installing

```bash
# Build the Rust binary
cd rust
cargo build --release

# For Python bindings (from root directory)
pip install -e .

# Run tests after installation
python test_install.py
```

### Running the Theorem Prover

```bash
# From rust directory
cargo run --bin prove -- <tptp_file> [options]

# Or use the release binary
./target/release/prove <tptp_file> [options]

# Common options:
#   --timeout <seconds>     Set timeout (default: 300s)
#   --max-clauses <n>       Set clause limit (default: 10000)
```

### Running Tests

#### Rust Tests
```bash
cd rust

# Run all tests
cargo test

# Run specific module tests
cargo test core
cargo test inference  
cargo test saturation
cargo test unification

# Run integration tests only
cargo test --test '*'

# Run with output for debugging
cargo test -- --nocapture
```

#### Python Tests
```bash
cd python
python -m pytest tests/ -v
```

### Test Files Location
- Rust unit tests: Colocated with source files
- Rust integration tests: `rust/tests/`
- Python tests: `python/tests/`
- TPTP test problems: `.tptp/TPTP-v9.0.0/Problems/`


## Codebase Architecture

### Rust Implementation

The core theorem prover is implemented in Rust for performance:

#### Core Module (`rust/src/core/`)
- **term.rs**: Terms (variables, constants, functions)
- **literal.rs**: Literals (positive/negative atoms)
- **clause.rs**: Clauses (disjunction of literals)
- **substitution.rs**: Variable substitutions with eager propagation
- **ordering.rs**: Knuth-Bendix Ordering (KBO) for term ordering
- **proof.rs**: Proof tracking and inference representation

#### Inference Module (`rust/src/inference/`)
- **resolution.rs**: Binary resolution
- **factoring.rs**: Factoring rule
- **superposition.rs**: Superposition calculus for equality
- **equality_resolution.rs**: Equality resolution (reflexivity)
- **equality_factoring.rs**: Equality factoring
- **demodulation.rs**: Demodulation (simplifying inference)
- **common.rs**: Shared utilities for inference rules

#### Saturation Module (`rust/src/saturation/`)
- **state.rs**: Saturation state (processed/unprocessed clauses)
- **subsumption.rs**: Forward/backward subsumption
- **simplification.rs**: Clause simplification (TODO)

#### Other Modules
- **parser/**: TPTP parser with CNF conversion
- **unification/**: Most General Unifier (MGU) computation
- **selection/**: Literal selection strategies

### Python Bindings

The Python interface is provided via PyO3 bindings:
- Compiled extension module: `proofatlas.cpython-*.so`
- Simple interface in `python/proofatlas/`
- Examples in `python/examples/`

### Project Structure

```
proofatlas/
├── rust/src/                  # Core Rust theorem prover
│   ├── core/                  # Terms, literals, clauses, substitutions, KBO
│   ├── inference/             # Resolution, factoring, superposition, demodulation
│   ├── saturation/            # Saturation loop, subsumption
│   ├── parser/                # TPTP parser, CNF conversion
│   ├── unification/           # MGU computation
│   ├── selectors/             # Clause/literal selection (Rust/Burn)
│   └── ml/                    # Graph building, inference
│
├── python/proofatlas/         # Python bindings and ML
│   ├── ml/                    # Config, training infrastructure
│   └── selectors/             # PyTorch model implementations
│
├── configs/
│   ├── data/                  # Data collection configs (Rust selector + problem filters)
│   └── training/              # ML training configs (gcn, gat, mlp, transformer)
│
├── scripts/                   # train.py, collect_data.py, compare_with_vampire.py
├── wasm/                      # WebAssembly frontend (index.html, models/)
├── docs/                      # Documentation
│
│ # External tools (gitignored):
├── .vampire/                  # Vampire theorem prover binary
├── .spass/                    # SPASS theorem prover
│
│ # Generated/data (gitignored):
├── .weights/                  # Trained model weights (safetensors)
├── .data/                     # TPTP problems, traces, problem_metadata.json
└── .logs/                     # Training logs and checkpoints
```

### Selector Architecture

ML selectors are implemented in both Rust/Burn (inference) and Python/PyTorch (training):

| Selector | Rust/Burn | PyTorch | Notes |
|----------|-----------|---------|-------|
| age_weight | ✓ | - | Heuristic only, no training needed |
| gcn | ✓ | ✓ | Graph Convolutional Network |
| mlp | ✓ | ✓ | MLP baseline |

**Workflow:** Train in PyTorch → Export to safetensors → Load in Rust/Burn

**Config types:**
- `configs/data/`: Rust selector for data collection
- `configs/training/`: PyTorch model architecture and hyperparameters


### Key Implementation Notes

1. **Unification with Eager Substitution**: The MGU algorithm uses eager substitution propagation to prevent unsound inferences. When adding a new substitution, all existing substitutions are immediately updated.

2. **Variable Renaming**: Variables from different clauses are renamed to avoid capture (e.g., X becomes X_c10 for clause 10).

3. **Superposition Calculus**: 
   - Only works on positive equalities
   - Respects term ordering (KBO)
   - Requires proper literal collection from parent clauses

4. **Literal Selection**: Four strategies from Hoder et al. "Selecting the selection" (2016):
   - Selection 0: Select all literals (default)
   - Selection 20: Select all maximal literals
   - Selection 21: Unique maximal, else neg max-weight, else all maximal
   - Selection 22: Neg max-weight, else all maximal

5. **Subsumption**: Both forward and backward subsumption are implemented with a pragmatic tiered approach.

6. **Proof Tracking**: Each inference stores premises, rule used, and the derived clause.

7. **Demodulation**: 
   - Uses one-way matching (only variables in the rewrite rule can be substituted)
   - The ordering constraint lσ ≻ rσ is strictly enforced
   - Applied in two contexts:
     - **Forward demodulation**: All new clauses are demodulated before being added to the clause set
     - **Backward demodulation**: When a unit equality is selected as given clause, all existing clauses are demodulated
   - This aggressive demodulation strategy is particularly effective for equational reasoning


### Debugging the Prover

```bash
# Debug specific clauses
cargo run --bin debug_ordering -- "term1" "term2"

# Save proof to file
cargo run --bin prove -- problem.p > proof.txt 2>&1

# Run with verbose output
RUST_LOG=debug cargo run --bin prove -- problem.p
```

### Working with the Codebase

When implementing new features:
1. Add unit tests alongside the implementation
2. Ensure variable renaming is consistent to avoid capture
3. Test with both satisfiable and unsatisfiable problems
4. Verify soundness - no proofs for satisfiable problems!
5. Use the prove binary for integration testing

### Recent Bug Fixes

1. **Superposition Literal Collection**: Fixed to ensure literals from the correct parent clauses are included in derived clauses.

2. **MGU Substitution Propagation**: Changed from lazy to eager propagation to prevent unsound variable substitutions.

3. **Proof Reporting**: Fixed clause index display to show actual indices instead of array positions.

4. **Demodulation Matching**: Fixed critical bug where unification was used instead of one-way matching. This caused unsound demodulations like reducing `mult(inv(Y),mult(Y,Z))` to `e` using pattern `mult(inv(X),X) = e`.

5. **Backward Demodulation**: Implemented backward demodulation when unit equalities are selected as given clauses. This dramatically improves performance on equational problems (e.g., uniqueness_of_inverse test went from timeout to 0.013s).

### Important: Analysis Guidelines

When a proof search times out or takes many steps, DO NOT conclude that "the proof is difficult" or make similar assessments. Instead, ask the user for analysis of what might be happening. The issue could be:
- A bug in the implementation
- Wrong selection strategy
- Missing inference rules
- Incorrect problem formulation
- Or many other factors

Always seek user input for analysis rather than making assumptions about proof difficulty.

### Important: File System Usage

When creating temporary files for testing or debugging:
- Always create them in the current working directory (not in /tmp)
- Clean up temporary files when done
- Add useful test files to the appropriate tests/ or test_traces/ directories

## Machine Learning Configuration

The ML pipeline uses JSON configuration files stored in `configs/`.

### Configuration Structure

```
configs/
├── data/           # Data collection configs (specify Rust selectors)
│   ├── default.json
│   ├── test.json
│   └── unit_equality.json
└── training/       # Training configs (specify PyTorch model architecture)
    ├── gcn.json
    ├── gat.json
    ├── mlp.json
    └── transformer.json
```

### Data Configuration (`configs/data/*.json`)

Controls which problems to use, which selector to run, and how to process data:

```json
{
  "name": "default",
  "description": "Default data configuration",

  "selector": {
    "name": "age_weight",           // Rust selector name
    "weights": null                  // Weights file in .weights/ (for ML selectors)
  },

  "solver": {
    "literal_selection": "0"  // 0=all, 20=maximal, 21=unique, 22=neg max-weight
  },

  "problem_filters": {
    "status": ["unsatisfiable"],
    "format": ["cnf"],
    "has_equality": null,
    "is_unit_only": null,
    "max_rating": 0.8,
    "min_clauses": 1,
    "max_clauses": 1000,
    "domains": null,
    "exclude_domains": ["CSR", "HWV", "SWV"]
  },

  "trace_collection": {
    "prover_timeout": 60.0,
    "max_clauses": 5000,
    "max_steps": 10000
  },

  "output": {
    "trace_dir": ".data/traces",
    "cache_dir": ".data/cache"
  }
}
```

### Training Configuration (`configs/training/*.json`)

Controls PyTorch model architecture and training hyperparameters:

```json
{
  "name": "gcn",
  "description": "Graph Convolutional Network clause selector",

  "model": {
    "type": "gcn",
    "hidden_dim": 64,
    "num_layers": 3,
    "num_heads": 4,
    "dropout": 0.1,
    "input_dim": 13
  },

  "training": {
    "data_config": "default",
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "max_epochs": 100,
    "patience": 10,
    "gradient_clip": 1.0
  },

  "optimizer": {
    "type": "adamw",
    "betas": [0.9, 0.999]
  },

  "scheduler": {
    "type": "cosine",
    "min_lr_ratio": 0.01,
    "warmup_epochs": 5
  }
}
```

### Using Configurations in Python

```python
from proofatlas.ml import DataConfig, TrainingConfig, list_configs

# List available presets
print(list_configs())
# {'data': ['default', 'test', 'unit_equality'], 'training': ['gcn', 'gat', 'mlp', 'transformer']}

# Load preset by name
data_cfg = DataConfig.load_preset("default")
train_cfg = TrainingConfig.load_preset("gcn")

# Access config values
print(data_cfg.selector.name)         # "age_weight"
print(train_cfg.model.type)           # "gcn"
print(train_cfg.training.batch_size)  # 32

# Save config
train_cfg.save("configs/training/my_experiment.json")
```

### Training Workflow

```bash
# 1. Extract problem metadata (one-time)
python scripts/extract_problem_metadata.py

# 2. Collect training data using a selector
python scripts/collect_data.py --data-config default --max-problems 100

# 3. Train a model and export to safetensors
python scripts/train.py --data .data/traces/default_data.pt --training gcn

# 4. Use trained model in data config:
# "selector": {"name": "gcn", "weights": "gcn.safetensors"}
```

### Problem Metadata

Problem metadata is extracted from TPTP and stored in `.data/problem_metadata.json`:

```bash
python scripts/extract_problem_metadata.py
```

This creates a JSON file with metadata for all CNF/FOF problems:
- path, domain, status (unsatisfiable/satisfiable/unknown)
- format (cnf/fof), has_equality, is_unit_only
- rating, num_clauses, num_axioms, num_conjectures
- num_predicates, num_functions, num_constants, max_term_depth