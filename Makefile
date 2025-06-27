# ProofAtlas Makefile for common development tasks

.PHONY: help setup test test-python test-rust build build-rust clean format lint

# Default target
help:
	@echo "ProofAtlas Development Commands"
	@echo "=============================="
	@echo "  make setup        - Set up conda environment and install dependencies"
	@echo "  make test         - Run all tests (Python + Rust)"
	@echo "  make test-python  - Run Python tests only"
	@echo "  make test-rust    - Run Rust tests only"
	@echo "  make build        - Build Rust components with maturin"
	@echo "  make build-rust   - Build Rust components with cargo"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make format       - Format Python and Rust code"
	@echo "  make lint         - Run linters on Python and Rust code"

# Setup environment
setup:
	./setup.sh

# Run all tests
test: test-python test-rust

# Run Python tests
test-python:
	cd python && python -m pytest tests/ -v

# Run Rust tests
test-rust:
	cd rust && cargo test

# Build Rust components with Python bindings
build:
	@command -v maturin >/dev/null 2>&1 || { echo "Error: maturin not found. Please install it with: pip install maturin"; exit 1; }
	maturin develop

# Build Rust components only
build-rust:
	cd rust && cargo build --release

# Clean build artifacts
clean:
	rm -rf rust/target
	rm -rf python/build
	rm -rf python/dist
	rm -rf python/src/*.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.dylib" -delete

# Format code
format:
	cd python && black src/ tests/ scripts/
	cd python && isort src/ tests/ scripts/
	cd rust && cargo fmt

# Run linters
lint:
	cd python && ruff check src/
	cd python && flake8 src/
	cd rust && cargo clippy