# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProofAtlas is a high-performance theorem prover implemented in Rust with Python bindings. The core implementation is in Rust, providing a complete saturation-based theorem prover with superposition calculus for equality reasoning.

## Testing the Theorem Prover

When asked to test the theorem prover:
- Use actual TPTP problems from `.data/problems/tptp/TPTP-v9.0.0/Problems/`
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
#   --no-superposition      Disable superposition calculus
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
- TPTP test problems: `.data/problems/tptp/TPTP-v9.0.0/Problems/`


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

### Current Project Structure

```
proofatlas/
├── python/                    # Python bindings and examples
│   ├── proofatlas/
│   │   ├── __init__.py
│   │   └── proofatlas.cpython-*.so  # Compiled Rust extension
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_interface.py
│   └── examples/
│       ├── basic_usage.py
│       ├── group_theory.py
│       └── interactive_demo.py
│
├── rust/                     # Core Rust implementation
│   ├── src/
│   │   ├── core/            # Core data structures
│   │   │   ├── clause.rs
│   │   │   ├── literal.rs
│   │   │   ├── term.rs
│   │   │   ├── substitution.rs
│   │   │   ├── ordering.rs  # KBO ordering
│   │   │   ├── proof.rs
│   │   │   └── mod.rs
│   │   ├── inference/       # Inference rules
│   │   │   ├── resolution.rs
│   │   │   ├── factoring.rs
│   │   │   ├── superposition.rs
│   │   │   ├── equality_resolution.rs
│   │   │   ├── equality_factoring.rs
│   │   │   ├── common.rs
│   │   │   └── mod.rs
│   │   ├── saturation/      # Saturation loop
│   │   │   ├── state.rs
│   │   │   ├── subsumption.rs
│   │   │   ├── simplification.rs
│   │   │   └── mod.rs
│   │   ├── parser/          # TPTP parser
│   │   │   ├── tptp.rs
│   │   │   ├── cnf_conversion.rs
│   │   │   ├── fof.rs
│   │   │   └── mod.rs
│   │   ├── unification/     # Unification
│   │   │   ├── mgu.rs
│   │   │   └── mod.rs
│   │   ├── selection/       # Literal selection
│   │   │   ├── clause.rs
│   │   │   ├── literal.rs
│   │   │   ├── max_weight.rs
│   │   │   └── mod.rs
│   │   ├── lib.rs
│   │   └── python_bindings.rs  # PyO3 bindings
│   ├── tests/               # Integration tests
│   │   ├── basic_test.rs
│   │   ├── test_calculus_compliance.rs
│   │   └── test_selection_behavior.rs
│   └── bin/                 # Binary executables
│       ├── prove.rs         # Main prover binary
│       ├── debug_ordering.rs
│       └── test_grp001.rs
│
├── docs/                    # Documentation
│   ├── python_interface_design.md
│   └── python_proofstate_api.md
│
├── setup.py                 # Python package setup
├── requirements.txt         # Python dependencies
├── INSTALL.md              # Installation guide
└── CLAUDE.md               # This file
```


### Key Implementation Notes

1. **Unification with Eager Substitution**: The MGU algorithm uses eager substitution propagation to prevent unsound inferences. When adding a new substitution, all existing substitutions are immediately updated.

2. **Variable Renaming**: Variables from different clauses are renamed to avoid capture (e.g., X becomes X_c10 for clause 10).

3. **Superposition Calculus**: 
   - Only works on positive equalities
   - Respects term ordering (KBO)
   - Requires proper literal collection from parent clauses

4. **Literal Selection**: Multiple strategies available:
   - First literal (default)
   - Max weight literal
   - Custom selection functions

5. **Subsumption**: Forward subsumption is implemented, backward subsumption is TODO.

6. **Proof Tracking**: Each inference stores premises, rule used, and the derived clause.

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

### Important: Analysis Guidelines

When a proof search times out or takes many steps, DO NOT conclude that "the proof is difficult" or make similar assessments. Instead, ask the user for analysis of what might be happening. The issue could be:
- A bug in the implementation
- Wrong selection strategy
- Missing inference rules
- Incorrect problem formulation
- Or many other factors

Always seek user input for analysis rather than making assumptions about proof difficulty.