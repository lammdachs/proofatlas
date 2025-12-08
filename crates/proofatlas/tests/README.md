# Integration Tests

This directory contains **integration tests** for the ProofAtlas Rust crate. These tests verify the public API and test how different modules work together.

## Important Note

**This directory does NOT contain all Rust tests!** The majority of tests are unit tests located within the source files themselves.

## Test Organization

- **Unit Tests**: Located in `src/` files, testing implementation details
  - Example: `src/algorithms/rules_tests.rs` contains tests for inference rules
  - Run with: `cargo test --lib`

- **Integration Tests** (this directory): Testing the public API
  - Example: `integration_test.rs` tests multiple rules working together
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
cargo test --test integration_test
```

## Purpose of Integration Tests

Integration tests in this directory:
- Can only access the public API (no private items)
- Test cross-module functionality
- Verify the crate works as external users would use it
- Each file is compiled as a separate crate

## Current Tests

- `integration_test.rs`: Tests array-based theorem prover with saturation loop and inference rules