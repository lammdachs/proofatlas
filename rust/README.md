# ProofAtlas Rust Implementation

A high-performance automated theorem prover using the superposition calculus.

## Features

- **Superposition calculus**: Complete implementation including resolution, factoring, and paramodulation
- **Standard data structures**: Clean type-safe representation using Rust structs and enums
- **TPTP parser**: Supports CNF format input files
- **Given-clause algorithm**: Efficient saturation-based proof search
- **Subsumption checking**: Automatic removal of redundant clauses

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
- `--no-superposition`: Disable superposition inference rule
- `--verbose`: Show detailed progress and proof steps

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