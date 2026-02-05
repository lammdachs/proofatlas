#!/bin/bash
# Benchmark script for proofatlas - sets up libtorch environment and runs prover

export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH

# Build and install the package first
maturin develop --release > /dev/null 2>&1

# Run the prover with timing
time proofatlas "$@"
