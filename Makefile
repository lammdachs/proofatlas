# ProofAtlas Makefile for common development tasks

.PHONY: help setup test test-python test-rust build build-rust clean format lint install install-dev

# Default target
help:
	@echo "ProofAtlas Development Commands"
	@echo "=============================="
	@echo "  make install      - Install package in development mode"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make build        - Build Rust extension"
	@echo "  make test         - Run all tests"
	@echo "  make test-python  - Run Python tests only"
	@echo "  make test-rust    - Run Rust tests only"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make format       - Format Python and Rust code"
	@echo "  make lint         - Run linters"
	@echo "  make examples     - Run example scripts"

# Install package in development mode
install:
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -e ".[dev]"

# Build Rust extension
build:
	python setup.py build_ext --inplace

# Run all tests
test: test-python test-rust

# Run Python tests
test-python:
	pytest python/tests/ -v

# Run Rust tests
test-rust:
	cd rust && cargo test

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/
	rm -rf rust/target/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find python -name "*.so" -delete
	find python -name "*.pyd" -delete
	find python -name "*.dylib" -delete

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

# Run examples
examples:
	@echo "Running basic usage example..."
	python python/examples/basic_usage.py
	@echo "\nRunning group theory example..."
	python python/examples/group_theory.py

# Build distribution packages
dist: clean
	python -m build

# Test coverage
test-cov:
	pytest python/tests/ --cov=proofatlas --cov-report=html --cov-report=term