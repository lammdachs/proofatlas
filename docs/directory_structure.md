# Directory Structure Guide

ProofAtlas uses a clean, module-mirroring directory structure that keeps data organized according to its purpose.

## Directory Layout

The setup script creates the following directory structure:

```
proofatlas/
├── .data/              # All data files
│   ├── problems/       # Theorem proving problems
│   │   └── tptp/      # TPTP library (optional)
│   ├── proofs/        # Generated proofs
│   ├── datasets/      # Prepared datasets
│   └── cache/         # Temporary files
├── .logs/             # Log files
│   └── runs/          # Execution logs
└── .selectors/        # ML selector resources (if PyTorch installed)
    ├── models/        # Trained models
    └── configs/       # Selector configurations
```

## Directory Purposes

### `.data/`
Main data directory containing all problem instances, proofs, and datasets.

- **`problems/`**: Store theorem proving problems in various formats (TPTP, custom)
- **`proofs/`**: Save generated proofs for analysis and inspection
- **`datasets/`**: Prepared datasets for training or evaluation
- **`cache/`**: Temporary files and caches for performance

### `.logs/`
Logging directory for debugging and analysis.

- **`runs/`**: Execution logs from proof searches

### `.selectors/` (Optional)
Created only when PyTorch is installed for ML-based clause selection.

- **`models/`**: Trained GNN or other ML models
- **`configs/`**: Configuration files for different selector strategies

## Environment Variables

The setup creates the following environment variables in `.env`:

```bash
# Main data directory
DATA_DIR=./.data

# Data subdirectories
PROBLEMS_DIR=./.data/problems
PROOFS_DIR=./.data/proofs
DATASETS_DIR=./.data/datasets
CACHE_DIR=./.data/cache

# TPTP library location
TPTP_PATH=./.data/problems/tptp

# Logging
LOG_DIR=./.logs

# Selector directories (if PyTorch installed)
SELECTORS_DIR=./.selectors
SELECTOR_MODELS_DIR=./.selectors/models
SELECTOR_CONFIGS_DIR=./.selectors/configs
```

## Customizing Paths

You can override any path by editing the `.env` file or setting environment variables:

```bash
export DATA_DIR=/custom/data/path
export LOG_DIR=/custom/logs
python your_script.py  # Will use the custom paths
```

## Docker Configuration

For Docker deployments, mount volumes to the standard locations:

```dockerfile
# In your docker-compose.yml
volumes:
  - ./data:/app/.data
  - ./logs:/app/.logs
  - ./selectors:/app/.selectors  # If using ML
```

## Best Practices

1. **Keep data organized**: Use the provided subdirectories for their intended purposes
2. **Version control**: Add `.data/`, `.logs/`, and `.selectors/` to `.gitignore`
3. **Backup proofs**: The `.data/proofs/` directory contains valuable experimental results
4. **Clean cache**: Periodically clean `.data/cache/` to free up space

## Migrating from Old Structure

If you have an existing ProofAtlas installation with the old directory structure, you can migrate:

```bash
# Move existing data to new structure
mv .problems/* .data/problems/
mv checkpoints/* .selectors/models/  # If you have trained models
mv experiments/* .data/proofs/       # If these contain proofs
mv .logs/* .logs/runs/
```