# ProofAtlas Rust Implementation

A high-performance automated theorem prover using the superposition calculus.

## Features

- **Superposition calculus**: Complete implementation including resolution, factoring, and paramodulation
- **Demodulation**: Both forward and backward demodulation for efficient equational reasoning
- **Standard data structures**: Clean type-safe representation using Rust structs and enums
- **TPTP parser**: Supports CNF format input files with automatic equality orientation
- **Given-clause algorithm**: Efficient saturation-based proof search
- **Subsumption checking**: Pragmatic redundancy elimination with tiered approach
- **Automatic equality orientation**: Equalities are preprocessed to ensure larger terms appear on the left (according to KBO)
- **Selection strategies**: Configurable literal and clause selection for controlling search space

## Building

```bash
cargo build --release
```

## Usage

```bash
# Basic usage
cargo run --bin prove -- problem.tptp

# With options
cargo run --bin prove -- problem.tptp --timeout 120 --max-clauses 50000 --verbose

# Run tests
cargo test
```

## CLI Options

- `--timeout <seconds>`: Set timeout in seconds (default: 60)
- `--max-clauses <n>`: Set maximum number of clauses (default: 10000)
- `--verbose`: Show detailed progress and proof steps
- `--literal-selection <strategy>`: Set literal selection strategy (all, max-weight)
- `--clause-selection <strategy>`: Set clause selection strategy (age, size, age-weight)

## Architecture

The implementation uses a modular design with clear separation of concerns:

- `formula/`: Core FOL data structures and operations
  - `mod.rs`: Type definitions (Term, Literal, Clause, etc.)
  - `unification.rs`: Robinson unification algorithm
  - `inference.rs`: Inference rules implementation
  - `saturation.rs`: Given-clause saturation loop
  - `parser.rs`: TPTP format parser

## Example

```bash
# Create a simple problem
cat > example.tptp << EOF
cnf(p_a, axiom, p(a)).
cnf(p_implies_q, axiom, ~p(X) | q(X)).
cnf(not_q_a, negated_conjecture, ~q(a)).
EOF

# Prove it
cargo run --bin prove -- example.tptp --verbose
```

## Documentation

- [Selection Strategies](docs/selection_strategies.md) - Detailed guide to literal and clause selection
- [Calculus Quick Reference](docs/calculus_quick_reference.md) - Inference rules and their conditions
- [Subsumption](docs/subsumption.md) - Pragmatic redundancy elimination strategy