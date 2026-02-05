#!/bin/bash
# Test script for proofatlas - sets up libtorch environment and runs cargo test

export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH

cargo test "$@"
