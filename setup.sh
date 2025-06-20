#!/bin/bash

# Exit on error
set -e

echo "Setting up ProofAtlas environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create conda environment if it doesn't exist
ENV_NAME="proofatlas"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Activating it..."
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -n ${ENV_NAME} python=3.11 -y
fi

# Activate the environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch and related packages via conda
echo "Installing PyTorch and CUDA dependencies..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
conda install -y pyg -c pyg

# Install PyTorch Lightning and related
echo "Installing PyTorch Lightning ecosystem..."
conda install -y pytorch-lightning torchmetrics -c conda-forge

# Install scientific computing packages
echo "Installing scientific computing packages..."
conda install -y numpy pandas matplotlib scikit-learn scipy sympy networkx -c conda-forge

# Install Jupyter and IPython
echo "Installing Jupyter and development tools..."
conda install -y jupyter ipython ipykernel notebook jupyterlab -c conda-forge

# Install configuration and CLI tools
echo "Installing configuration and CLI tools..."
conda install -y omegaconf click pyyaml -c conda-forge

# Install parsing libraries
echo "Installing parsing libraries..."
conda install -y antlr-python-runtime lark-parser -c conda-forge

# Install additional utilities
echo "Installing additional utilities..."
conda install -y tqdm gitpython python-dotenv rich pip setuptools wheel -c conda-forge

# Install development tools
echo "Installing development tools..."
conda install -y pytest pytest-cov black isort flake8 mypy ruff ipdb -c conda-forge

# Install visualization tools
echo "Installing visualization tools..."
conda install -y seaborn plotly -c conda-forge

# Create environment.yml for easy environment recreation
echo "Creating environment.yml file..."
cat > environment.yml << 'EOF'
name: proofatlas
channels:
  - pytorch
  - nvidia
  - pyg
  - conda-forge
  - defaults
dependencies:
  # Python
  - python=3.11
  
  # PyTorch ecosystem
  - pytorch
  - pytorch-cuda=12.1
  - pyg
  - pytorch-lightning
  - torchmetrics
  
  # Scientific computing
  - numpy
  - matplotlib
  - scikit-learn
  - scipy
  - networkx
  - plotly
  
  # Jupyter ecosystem
  - jupyter
  - ipython
  - ipykernel
  - notebook
  
  # Configuration and CLI
  - omegaconf
  - click
  - pyyaml
  
  # Parsing
  - lark-parser
  
  # Utilities
  - tqdm
  - gitpython
  - python-dotenv
  - rich
  - textual
  - pip
  - setuptools
  - wheel
  
  # Development tools
  - pytest
  - pytest-cov
  - black
  - isort
  - flake8
  - ruff
EOF

# Export current environment
echo "Exporting current environment state..."
conda env export --no-builds > environment.lock.yml

# Install the package in development mode
echo "Installing ProofAtlas package in development mode..."
pip install -e . --no-deps

# Create or update .env file
echo ""
echo "Configuring environment variables..."
echo ""

# Function to read input with a default value
read_with_default() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    echo -n "$prompt [$default]: "
    read -r input_value
    
    if [ -z "$input_value" ]; then
        eval "$var_name='$default'"
    else
        eval "$var_name='$input_value'"
    fi
}

# Function to read sensitive input (hidden)
read_sensitive() {
    local prompt="$1"
    local var_name="$2"
    
    echo -n "$prompt: "
    read -s -r input_value
    echo ""
    
    eval "$var_name='$input_value'"
}

# Check if .env exists and ask about overwriting
if [ -f .env ]; then
    echo "An .env file already exists."
    echo -n "Do you want to update it? (y/N): "
    read -r update_env
    
    if [ "$update_env" != "y" ] && [ "$update_env" != "Y" ]; then
        echo "Keeping existing .env file."
    else
        # Backup existing .env
        cp .env .env.backup
        echo "Backed up existing .env to .env.backup"
        
        # Read configuration values
        echo ""
        echo "=== Data Paths Configuration ==="
        read_with_default "Enter data directory path" "./.problems" DATA_DIR
        read_with_default "Enter cache directory path" "./cache" CACHE_DIR
        read_with_default "Enter TPTP library path" "./.problems/tptp" TPTP_PATH
        
        echo ""
        echo "=== Model Paths Configuration ==="
        read_with_default "Enter models directory path" "./.models" MODEL_DIR
        read_with_default "Enter checkpoints directory path" "./checkpoints" CHECKPOINT_DIR
        
        echo ""
        echo "=== Experiment Configuration ==="
        read_with_default "Enter experiments directory path" "./experiments" EXPERIMENT_DIR
        read_with_default "Enter logs directory path" "./.logs" LOG_DIR
        
        # Write .env file
        cat > .env << EOF
# Environment variables for ProofAtlas
# Generated on $(date)

# Data paths
DATA_DIR=${DATA_DIR}
CACHE_DIR=${CACHE_DIR}
TPTP_PATH=${TPTP_PATH}

# Model paths
MODEL_DIR=${MODEL_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR}

# Experiment configuration
EXPERIMENT_DIR=${EXPERIMENT_DIR}
LOG_DIR=${LOG_DIR}

# Additional environment variables can be added here
EOF
        
        echo ""
        echo ".env file updated successfully!"
    fi
else
    # Create new .env file
    echo "No .env file found. Creating new configuration..."
    
    # Read configuration values
    echo ""
    echo "=== Data Paths Configuration ==="
    read_with_default "Enter data directory path" "./.problems" DATA_DIR
    read_with_default "Enter cache directory path" "./cache" CACHE_DIR
    read_with_default "Enter TPTP library path" "./.problems/tptp" TPTP_PATH
    
    echo ""
    echo "=== Model Paths Configuration ==="
    read_with_default "Enter models directory path" "./.models" MODEL_DIR
    read_with_default "Enter checkpoints directory path" "./checkpoints" CHECKPOINT_DIR
    
    echo ""
    echo "=== Experiment Configuration ==="
    read_with_default "Enter experiments directory path" "./experiments" EXPERIMENT_DIR
    read_with_default "Enter logs directory path" "./.logs" LOG_DIR
    
    # Write .env file
    cat > .env << EOF
# Environment variables for ProofAtlas
# Generated on $(date)

# Data paths
DATA_DIR=${DATA_DIR}
CACHE_DIR=${CACHE_DIR}
TPTP_PATH=${TPTP_PATH}

# Model paths
MODEL_DIR=${MODEL_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR}

# Experiment configuration
EXPERIMENT_DIR=${EXPERIMENT_DIR}
LOG_DIR=${LOG_DIR}

# Additional environment variables can be added here
EOF
    
    echo ""
    echo ".env file created successfully!"
fi

# Set proper permissions for .env file
chmod 600 .env

# Create necessary directories based on configuration
echo "Creating project directories..."
# Source the .env file to get the directory paths
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Create directories, expanding ~ if present
mkdir -p "${DATA_DIR/#\~/$HOME}" "${CACHE_DIR/#\~/$HOME}" "${MODEL_DIR/#\~/$HOME}" "${CHECKPOINT_DIR/#\~/$HOME}" "${EXPERIMENT_DIR/#\~/$HOME}" "${LOG_DIR/#\~/$HOME}"

# Ask about TPTP library download
echo ""
echo "=== TPTP Library Setup ==="
echo "The TPTP (Thousands of Problems for Theorem Provers) library contains"
echo "a comprehensive collection of theorem proving problems."
echo ""
echo -n "Would you like to download and set up the TPTP library? (y/N): "
read -r download_tptp

if [ "$download_tptp" = "y" ] || [ "$download_tptp" = "Y" ]; then
    echo ""
    echo "Downloading TPTP library..."
    
    # Expand TPTP_PATH if it contains ~
    TPTP_PATH_EXPANDED="${TPTP_PATH/#\~/$HOME}"
    
    # Create TPTP directory
    mkdir -p "$TPTP_PATH_EXPANDED"
    
    # Download TPTP archive
    TPTP_URL="http://www.tptp.org/TPTP/Distribution/TPTP-v8.2.0.tgz"
    TPTP_ARCHIVE="$TPTP_PATH_EXPANDED/TPTP-v8.2.0.tgz"
    
    if [ -f "$TPTP_ARCHIVE" ]; then
        echo "TPTP archive already exists. Skipping download."
    else
        echo "Downloading from $TPTP_URL..."
        curl -L -o "$TPTP_ARCHIVE" "$TPTP_URL" || wget -O "$TPTP_ARCHIVE" "$TPTP_URL"
    fi
    
    # Extract TPTP archive
    echo "Extracting TPTP archive..."
    cd "$TPTP_PATH_EXPANDED"
    tar -xzf "TPTP-v8.2.0.tgz"
    
    # Create symlink to Problems directory
    if [ -d "TPTP-v8.2.0/Problems" ]; then
        ln -sf "TPTP-v8.2.0/Problems" "Problems"
        echo "TPTP library extracted successfully!"
        echo "Problems directory: $TPTP_PATH_EXPANDED/Problems"
    else
        echo "Warning: Problems directory not found in TPTP archive."
    fi
    
    cd "$SCRIPT_DIR"
    
    # Optionally clean up archive
    echo -n "Would you like to keep the downloaded archive? (y/N): "
    read -r keep_archive
    if [ "$keep_archive" != "y" ] && [ "$keep_archive" != "Y" ]; then
        rm -f "$TPTP_ARCHIVE"
        echo "Archive removed."
    fi
else
    echo "Skipping TPTP library download."
    echo "You can download it manually later from: http://www.tptp.org/"
fi

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f .pre-commit-config.yaml ]; then
    echo "Installing pre-commit hooks..."
    conda install -y pre-commit -c conda-forge
    pre-commit install
fi

echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To recreate this environment on another machine, run:"
echo "  conda env create -f environment.yml"
echo ""
echo "To verify the installation, run:"
echo '  python -c "import torch; import torch_geometric; print(f'\''PyTorch: {torch.__version__}'\''); print(f'\''PyG: {torch_geometric.__version__}'\''); print(f'\''CUDA available: {torch.cuda.is_available()}'\'')"'
echo ""

# Display current configuration
if [ -f .env ]; then
    echo "Current configuration (.env):"
    echo "============================="
    grep -E "^[A-Z_]+=" .env | while IFS='=' read -r key value; do
        echo "  $key: $value"
    done
    echo ""
fi

echo "Next steps:"
echo "  1. Activate the environment: conda activate ${ENV_NAME}"
echo "  2. Download any required datasets to the ${DATA_DIR} directory"
echo "  3. Start developing with ProofAtlas!"