from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="proofatlas",
    version="0.2.0",
    description="High-performance theorem prover for first-order logic",
    author="ProofAtlas Contributors",
    author_email="",
    url="https://github.com/lexpk/proofatlas",
    
    # Find packages in the python/ directory
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    
    # Rust extension configuration
    rust_extensions=[
        RustExtension(
            "proofatlas.proofatlas",
            path="rust/Cargo.toml",
            binding=Binding.PyO3,
            features=["python"],
        )
    ],
    
    zip_safe=False,
    python_requires=">=3.7",
    
    install_requires=[
        # Core dependencies can be added here
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "mypy",
            "types-setuptools",
            "ruff",
        ],
        "examples": [
            "matplotlib>=3.0",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
        "License :: OSI Approved :: MIT License",
    ],
    
    # Entry points if needed
    entry_points={
        # 'console_scripts': [
        #     'proofatlas=proofatlas.cli:main',
        # ],
    },
)