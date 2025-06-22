# Configuration Profiles Guide

ProofAtlas supports multiple configuration profiles to accommodate different use cases and environments. During setup, you can choose from predefined profiles or create a custom configuration.

## Available Profiles

### 1. Local Profile (Default)

All data and resources are stored within the project directory.

```
proofatlas/
├── .problems/          # Data directory
│   └── tptp/          # TPTP library
├── .models/           # Trained models
├── checkpoints/       # Training checkpoints
├── experiments/       # Experiment results
├── .logs/            # Log files
└── cache/            # Cache directory
```

**Best for:**
- Personal development
- Isolated projects
- Quick testing
- CI/CD environments

**Example .env:**
```bash
DATA_DIR=./.problems
CACHE_DIR=./cache
TPTP_PATH=./.problems/tptp
MODEL_DIR=./.models
CHECKPOINT_DIR=./checkpoints
EXPERIMENT_DIR=./experiments
LOG_DIR=./.logs
```

### 2. Centralized Profile

Shares data and models across multiple projects while keeping experiment-specific files local.

```
~/theorem-proving/      # Shared resources
├── data/
│   ├── tptp/          # Shared TPTP library
│   └── other/         # Other datasets
├── models/
│   └── proofatlas/    # Models for this project
└── cache/             # Shared cache

proofatlas/            # Project-specific
├── checkpoints/       # Local checkpoints
├── experiments/       # Local experiments
└── logs/             # Local logs
```

**Best for:**
- Multiple theorem proving projects
- Sharing TPTP library (it's large)
- Model reuse across projects
- Collaborative research

**Example .env:**
```bash
DATA_DIR=~/theorem-proving/data
CACHE_DIR=~/theorem-proving/cache
TPTP_PATH=~/theorem-proving/data/tptp
MODEL_DIR=~/theorem-proving/models/proofatlas
CHECKPOINT_DIR=./checkpoints
EXPERIMENT_DIR=./experiments
LOG_DIR=./logs
```

### 3. HPC/Cluster Profile

Optimized for high-performance computing environments with separate fast and persistent storage.

```
/scratch/$USER/         # Fast SSD storage
├── theorem-proving/
│   ├── data/          # Large datasets
│   └── cache/         # Temporary cache
└── proofatlas/
    ├── models/        # Working models
    └── checkpoints/   # Active checkpoints

/home/$USER/proofatlas/ # Persistent storage
├── experiments/        # Results to keep
└── logs/              # All logs
```

**Best for:**
- Large-scale experiments
- Cluster environments (SLURM, PBS, LSF)
- Training with large datasets
- Distributed computing

**Example .env:**
```bash
DATA_DIR=/scratch/$USER/theorem-proving/data
CACHE_DIR=/scratch/$USER/theorem-proving/cache
TPTP_PATH=/scratch/$USER/theorem-proving/data/tptp
MODEL_DIR=/scratch/$USER/proofatlas/models
CHECKPOINT_DIR=/scratch/$USER/proofatlas/checkpoints
EXPERIMENT_DIR=/home/$USER/proofatlas/experiments
LOG_DIR=/home/$USER/proofatlas/logs
```

### 4. Custom Profile

Manually specify each path for specific requirements.

**Best for:**
- Existing directory structures
- Special requirements
- Docker/container setups
- Network storage configurations

## Automatic HPC Detection

The setup script automatically detects common HPC environments:
- SLURM (checks for `SLURM_JOB_ID` or `sbatch`)
- PBS (checks for `PBS_JOBID` or `qsub`)
- LSF (checks for `LSB_JOBID` or `bsub`)

If detected, it will suggest using the HPC profile.

## Changing Profiles

To change your configuration profile:

1. Run the setup script again:
   ```bash
   ./setup.sh
   ```

2. When prompted about updating .env, choose "yes"

3. Select your new profile

4. Your existing .env will be backed up to .env.backup

## Environment Variables

You can override any path by setting environment variables:

```bash
export DATA_DIR=/custom/data/path
export MODEL_DIR=/custom/model/path
python train.py  # Will use the custom paths
```

## Docker Configuration

For Docker deployments, use the custom profile with mounted volumes:

```dockerfile
# In your Dockerfile or docker-compose.yml
volumes:
  - ./data:/data
  - ./models:/models
  - ./experiments:/experiments

# In .env
DATA_DIR=/data
MODEL_DIR=/models
EXPERIMENT_DIR=/experiments
```

## Best Practices

1. **Use Centralized for Multiple Projects**: Share large datasets like TPTP across projects

2. **Use HPC for Training**: Put training data on fast scratch storage, keep results on persistent storage

3. **Version Control .env**: Add .env to .gitignore but keep .env.example with your profile choice

4. **Document Your Setup**: Note which profile you used in your project README

5. **Backup Important Data**: Especially on HPC systems where scratch is often purged

## Troubleshooting

### Permission Issues
- Ensure you have write permissions for all configured directories
- On shared systems, check group permissions for shared directories

### Path Not Found
- The setup script creates directories automatically
- For custom paths, ensure parent directories exist

### HPC Scratch Purge
- Many HPC systems purge scratch directories after 30-90 days
- Keep important results in persistent storage (home directory)

### Network Storage
- For NFS/network paths, ensure they're mounted before running
- Consider using local cache for better performance