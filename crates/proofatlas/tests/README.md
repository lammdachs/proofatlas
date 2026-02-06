# Integration Tests

This directory contains **integration tests** for the ProofAtlas Rust crate. These tests verify the public API and test how different modules work together.

## Important Note

**This directory does NOT contain all Rust tests!** The majority of tests are unit tests located within the source files themselves.

## Test Organization

- **Unit Tests**: Located in `src/` files, testing implementation details
  - Run with: `cargo test --lib`

- **Integration Tests** (this directory): Testing the public API
  - Run with: `cargo test --test '*'`

## Running Tests

```bash
# Run ALL tests (unit + integration)
cargo test

# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test '*'

# Run a specific integration test file
cargo test --test basic_test
```

## Current Tests

- `basic_test.rs`: Basic theorem prover functionality
- `test_calculus_compliance.rs`: Superposition calculus correctness
- `test_group_theory.rs`: Group theory problem solving
- `test_krs065_bug.rs`: Regression test for KRS065 bug
- `test_literal_selection.rs`: Literal selection strategies
- `test_selection_behavior.rs`: Clause selection behavior
