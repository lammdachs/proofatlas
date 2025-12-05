# ProofAtlas Makefile for common development tasks

.PHONY: help install install-dev build build-release test test-python test-rust clean format lint bench

# Default target
help:
	@echo "ProofAtlas Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install package in development mode"
	@echo "  make install-dev  - Install with development dependencies"
	@echo ""
	@echo "Build:"
	@echo "  make build        - Build Rust prover (debug)"
	@echo "  make build-release - Build Rust prover (release)"
	@echo ""
	@echo "Test:"
	@echo "  make test         - Run all tests"
	@echo "  make test-python  - Run Python tests only"
	@echo "  make test-rust    - Run Rust tests only"
	@echo ""
	@echo "Quality:"
	@echo "  make format       - Format Python and Rust code"
	@echo "  make lint         - Run linters"
	@echo ""
	@echo "Benchmark:"
	@echo "  make bench        - Run benchmarks (proofatlas-bench)"
	@echo ""
	@echo "Other:"
	@echo "  make clean        - Clean build artifacts"

# Install package in development mode
install:
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -e ".[dev]"

# Build Rust prover (debug)
build:
	cd rust && cargo build

# Build Rust prover (release)
build-release:
	cd rust && cargo build --release

# Run all tests
test: test-rust test-python

# Run Python tests
test-python:
	python -m pytest python/tests/ -v

# Run Rust tests
test-rust:
	cd rust && cargo test

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/
	rm -rf rust/target/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find python -name "*.so" -delete 2>/dev/null || true
	find python -name "*.pyd" -delete 2>/dev/null || true
	find python -name "*.dylib" -delete 2>/dev/null || true

# Format code
format:
	black python/
	cd rust && cargo fmt

# Run linters
lint:
	ruff check python/
	black --check python/
	cd rust && cargo clippy

# Type check Python code
type-check:
	mypy python/

# Run benchmarks
bench:
	proofatlas-bench --track

# Run prover on a problem
prove:
	@echo "Usage: make prove PROBLEM=path/to/problem.p"
	@echo "Example: make prove PROBLEM=.tptp/TPTP-v9.0.0/Problems/PUZ/PUZ001-1.p"
ifdef PROBLEM
	./rust/target/release/prove $(PROBLEM)
endif
