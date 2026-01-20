#!/usr/bin/env python3
"""
Setup script for proofatlas.

This script sets required environment variables for tch-rs/libtorch
before invoking the standard setuptools build.
"""
import os
import sys

# Set environment variable for tch-rs to find PyTorch's libtorch
os.environ.setdefault('LIBTORCH_USE_PYTORCH', '1')

# Import and run setuptools
from setuptools import setup

if __name__ == '__main__':
    setup()
