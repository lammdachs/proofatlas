#!/bin/bash

# Exit on error
set -e

echo "Setting up ProofAtlas environment..."

# Function to ask yes/no questions
ask_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    
    if [ "$default" = "y" ]; then
        prompt="$prompt (Y/n): "
    else
        prompt="$prompt (y/N): "
    fi
    
    echo -n "$prompt"
    read -r response
    
    if [ -z "$response" ]; then
        response="$default"
    fi
    
    [[ "$response" =~ ^[Yy]$ ]]
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Conda (Anaconda or Miniconda) is required for ProofAtlas."
    echo ""
    echo "Miniconda is a minimal conda installer that includes only conda, Python, and a few other packages."
    echo "It's perfect for ProofAtlas and takes up less space than the full Anaconda distribution."
    echo ""
    
    if ask_yes_no "Would you like to install Miniconda now?"; then
        echo ""
        echo "Installing Miniconda..."
        
        # Detect OS and architecture
        OS_TYPE=$(uname -s)
        ARCH_TYPE=$(uname -m)
        
        # Determine the appropriate Miniconda installer
        if [ "$OS_TYPE" = "Linux" ]; then
            if [ "$ARCH_TYPE" = "x86_64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            elif [ "$ARCH_TYPE" = "aarch64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
            else
                echo "Error: Unsupported Linux architecture: $ARCH_TYPE"
                exit 1
            fi
        elif [ "$OS_TYPE" = "Darwin" ]; then
            if [ "$ARCH_TYPE" = "x86_64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            elif [ "$ARCH_TYPE" = "arm64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            else
                echo "Error: Unsupported macOS architecture: $ARCH_TYPE"
                exit 1
            fi
        else
            echo "Error: Unsupported operating system: $OS_TYPE"
            echo "Please install Miniconda manually from: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
        
        # Download Miniconda installer
        MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
        echo "Downloading Miniconda from: $MINICONDA_URL"
        if curl -L -o "$MINICONDA_INSTALLER" "$MINICONDA_URL" || wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"; then
            # Make installer executable
            chmod +x "$MINICONDA_INSTALLER"
            
            # Determine installation directory
            echo ""
            echo "Where would you like to install Miniconda?"
            echo "Press Enter for default location ($HOME/miniconda3),"
            echo "or enter a custom path:"
            read -r CONDA_PREFIX_PATH
            
            if [ -z "$CONDA_PREFIX_PATH" ]; then
                CONDA_PREFIX_PATH="$HOME/miniconda3"
            fi
            
            # Run installer
            echo "Installing to: $CONDA_PREFIX_PATH"
            "$MINICONDA_INSTALLER" -b -p "$CONDA_PREFIX_PATH"
            
            # Clean up installer
            rm -f "$MINICONDA_INSTALLER"
            
            # Initialize conda for current shell
            echo ""
            echo "Initializing conda..."
            "$CONDA_PREFIX_PATH/bin/conda" init bash
            
            # Also init for zsh if it's installed
            if command -v zsh &> /dev/null; then
                "$CONDA_PREFIX_PATH/bin/conda" init zsh
            fi
            
            echo ""
            echo "Miniconda has been installed successfully!"
            echo ""
            echo "IMPORTANT: You need to restart your shell or run:"
            echo "  source ~/.bashrc  (for bash)"
            echo "  source ~/.zshrc   (for zsh)"
            echo ""
            echo "Then run this setup script again."
            echo ""
            
            # Ask if user wants to continue with sourcing
            if ask_yes_no "Would you like to source your shell config now and continue?" "y"; then
                if [ -n "$BASH_VERSION" ]; then
                    source ~/.bashrc
                elif [ -n "$ZSH_VERSION" ]; then
                    source ~/.zshrc
                else
                    echo "Please source your shell configuration manually and run setup again."
                    exit 0
                fi
                
                # Verify conda is now available
                if ! command -v conda &> /dev/null; then
                    echo "Error: conda still not found in PATH."
                    echo "Please restart your terminal and run this setup script again."
                    exit 1
                fi
            else
                echo "Please restart your terminal and run this setup script again."
                exit 0
            fi
        else
            echo "Error: Failed to download Miniconda installer."
            echo "Please install Miniconda manually from: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
    else
        echo ""
        echo "Please install Anaconda or Miniconda manually."
        echo ""
        echo "To install Miniconda:"
        echo "  1. Visit: https://docs.conda.io/en/latest/miniconda.html"
        echo "  2. Download the installer for your operating system"
        echo "  3. Run the installer"
        echo "  4. Restart your terminal"
        echo "  5. Run this setup script again"
        echo ""
        echo "To install Anaconda (larger, includes many packages):"
        echo "  Visit: https://www.anaconda.com/products/distribution"
        exit 1
    fi
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if environment.yml exists
if [ ! -f environment.yml ]; then
    echo "Error: environment.yml not found in the current directory."
    echo "Please ensure you're running this script from the ProofAtlas root directory."
    exit 1
fi

# Create or update conda environment from environment.yml
ENV_NAME="proofatlas"
echo ""
echo "Creating/updating conda environment '${ENV_NAME}' from environment.yml..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Updating it..."
    conda env update -n ${ENV_NAME} -f environment.yml --prune
else
    echo "Creating new environment '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

# Activate the environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch and Machine Learning packages (required)
echo ""
echo "=== Installing PyTorch and GNN Components ==="
echo ""
echo "PyTorch and Graph Neural Networks are required for ProofAtlas."
echo "These packages enable:"
echo "   - GNN-based clause selection"
echo "   - Learned proof guidance"
echo "   - Neural premise selection"
echo ""

echo "PyTorch can be installed with or without GPU support:"
echo "  1. CPU-only (smaller, works everywhere)"
echo "  2. GPU with CUDA (requires NVIDIA GPU and CUDA drivers)"
echo ""
echo "   IMPORTANT: Installing GPU components requires accepting the CUDA EULA:"
echo "   https://docs.nvidia.com/cuda/eula/index.html"
echo ""

GPU_SUPPORT="no"
CUDA_VERSION=""
CUDA_COMPILER_VERSION=""

echo "PyTorch Installation Options:"
echo "=============================="
echo ""

# Check if NVIDIA GPU is available and show driver info
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Current driver information:"
    nvidia-smi | grep "Driver Version" || true
    echo ""
fi

echo "Enter the CUDA version you want to install"
echo "Or press Enter for CPU-only installation:"
echo ""
echo "WARNING: Choose a CUDA version compatible with your GPU and drivers!"
echo "Note: Due to conda package resolution issues, you may need to specify"
echo "      additional version constraints (e.g., for CUDA 12.*, you might"
echo "      need cuda-compiler=12.4.1)"
echo ""
read -r -p "CUDA version (or press Enter for CPU-only): " cuda_input

if [ -z "$cuda_input" ]; then
    GPU_SUPPORT="no"
    echo "Selected CPU-only installation."
else
    CUDA_VERSION="$cuda_input"
    GPU_SUPPORT="yes"
    echo "Selected CUDA $CUDA_VERSION"
    
    # Ask about cuda-compiler version if needed
    echo ""
    read -r -p "Specify cuda-compiler version (or press Enter to use default): " cuda_compiler_input
    if [ -n "$cuda_compiler_input" ]; then
        CUDA_COMPILER_VERSION="$cuda_compiler_input"
        echo "Will install cuda-compiler=$CUDA_COMPILER_VERSION"
    fi
fi

if [ "$GPU_SUPPORT" = "yes" ]; then
    echo ""
    echo "CUDA $CUDA_VERSION will be installed via conda."
    echo "This includes the CUDA toolkit for the conda environment."
    echo ""
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: No NVIDIA GPU detected with nvidia-smi."
        echo "GPU acceleration requires an NVIDIA GPU with proper drivers."
        echo ""
    fi
    
    if ! ask_yes_no "Continue with CUDA $CUDA_VERSION installation?" "y"; then
        GPU_SUPPORT="no"
        echo "Switching to CPU-only installation."
    fi
fi

echo ""
echo "Installing PyTorch ecosystem..."

if [ "$GPU_SUPPORT" = "yes" ]; then
    echo "Installing CUDA toolkit $CUDA_VERSION..."
    if [ -n "$CUDA_COMPILER_VERSION" ]; then
        conda install -y cuda-toolkit=$CUDA_VERSION cuda-compiler=$CUDA_COMPILER_VERSION -c nvidia
    else
        conda install -y cuda-toolkit=$CUDA_VERSION -c nvidia
    fi
    
    echo "Installing PyTorch with GPU support (CUDA $CUDA_VERSION)..."
    # Convert CUDA version format (e.g., 12.1 -> cu121)
    CUDA_VERSION_SHORT=$(echo $CUDA_VERSION | sed 's/\.//g')
    pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}
    
    echo "Installing PyTorch Geometric..."
    pip install torch-geometric
    
    echo "Installing PyG dependencies for GPU..."
    pip install pyg-lib -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu${CUDA_VERSION_SHORT}.html
else
    echo "Installing PyTorch CPU-only version..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    
    echo "Installing PyTorch Geometric..."
    pip install torch-geometric
    
    echo "Installing PyG dependencies for CPU..."
    pip install pyg-lib -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
fi

echo "Installing PyTorch Lightning..."
pip install pytorch-lightning torchmetrics

echo "PyTorch and GNN packages installed successfully!"
PYTORCH_INSTALLED="yes"

echo ""

# Install the package in development mode
echo ""
echo "Installing ProofAtlas package in development mode..."
cd python && pip install -e . --no-deps && cd ..

# Create or update .env file
echo ""
echo "Configuring environment variables..."
echo ""

# Set up simple module-mirroring directory structure
DATA_DIR="./.data"
PROBLEMS_DIR="./.data/problems"
PROOFS_DIR="./.data/proofs"
DATASETS_DIR="./.data/datasets"
CACHE_DIR="./.data/cache"
TPTP_PATH="./.data/problems/tptp"
LOG_DIR="./.logs"

# Only create selector directories if PyTorch is being installed
SELECTORS_DIR="./.selectors"
SELECTOR_MODELS_DIR="./.selectors/models"
SELECTOR_CONFIGS_DIR="./.selectors/configs"

# Check if .env exists and ask about overwriting
if [ -f .env ]; then
    echo "An .env file already exists."
    if ask_yes_no "Do you want to update it?" "n"; then
        # Backup existing .env
        cp .env .env.backup
        echo "Backed up existing .env to .env.backup"
        
        # Write .env file with new structure
        cat > .env << EOF
# Environment variables for ProofAtlas
# Generated on $(date)
# Module-mirroring directory structure

# Main data directory
DATA_DIR=${DATA_DIR}

# Data subdirectories
PROBLEMS_DIR=${PROBLEMS_DIR}
PROOFS_DIR=${PROOFS_DIR}
DATASETS_DIR=${DATASETS_DIR}
CACHE_DIR=${CACHE_DIR}

# TPTP library location
TPTP_PATH=${TPTP_PATH}

# Logging
LOG_DIR=${LOG_DIR}

# Selector directories (for ML models)
SELECTORS_DIR=${SELECTORS_DIR}
SELECTOR_MODELS_DIR=${SELECTOR_MODELS_DIR}
SELECTOR_CONFIGS_DIR=${SELECTOR_CONFIGS_DIR}

# Additional environment variables can be added here
EOF
        
        echo ""
        echo ".env file updated successfully!"
    else
        echo "Keeping existing .env file."
    fi
else
    # Create new .env file
    echo "No .env file found. Creating new configuration..."
    
    # Write .env file with new structure
    cat > .env << EOF
# Environment variables for ProofAtlas
# Generated on $(date)
# Module-mirroring directory structure

# Main data directory
DATA_DIR=${DATA_DIR}

# Data subdirectories
PROBLEMS_DIR=${PROBLEMS_DIR}
PROOFS_DIR=${PROOFS_DIR}
DATASETS_DIR=${DATASETS_DIR}
CACHE_DIR=${CACHE_DIR}

# TPTP library location
TPTP_PATH=${TPTP_PATH}

# Logging
LOG_DIR=${LOG_DIR}

# Selector directories (for ML models)
SELECTORS_DIR=${SELECTORS_DIR}
SELECTOR_MODELS_DIR=${SELECTOR_MODELS_DIR}
SELECTOR_CONFIGS_DIR=${SELECTOR_CONFIGS_DIR}

# Additional environment variables can be added here
EOF
    
    echo ""
    echo ".env file created successfully!"
fi

# Set proper permissions for .env file
chmod 600 .env

# Create necessary directories based on configuration
echo ""
echo "Creating project directories..."

# Source the .env file to get the directory paths
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Create main directories, expanding ~ if present
mkdir -p "${DATA_DIR/#\~/$HOME}"
mkdir -p "${PROBLEMS_DIR/#\~/$HOME}"
mkdir -p "${PROOFS_DIR/#\~/$HOME}"
mkdir -p "${DATASETS_DIR/#\~/$HOME}"
mkdir -p "${CACHE_DIR/#\~/$HOME}"
mkdir -p "${LOG_DIR/#\~/$HOME}/runs"

echo "Created directory structure:"
echo "  $DATA_DIR/"
echo "  ├── problems/    # Theorem proving problems"
echo "  ├── proofs/      # Generated proofs"
echo "  ├── datasets/    # Prepared datasets"
echo "  └── cache/       # Temporary files"
echo "  $LOG_DIR/        # Execution logs"

# Create selector directories (PyTorch is now always installed)
mkdir -p "${SELECTORS_DIR/#\~/$HOME}"
mkdir -p "${SELECTOR_MODELS_DIR/#\~/$HOME}"
mkdir -p "${SELECTOR_CONFIGS_DIR/#\~/$HOME}"
echo "  $SELECTORS_DIR/  # ML selector resources"
echo "  ├── models/      # Trained models"
echo "  └── configs/     # Configurations"

# Ask about TPTP library download
echo ""
echo "=== TPTP Library Setup ==="
echo "The TPTP (Thousands of Problems for Theorem Provers) library contains"
echo "a comprehensive collection of theorem proving problems."
echo ""

# Function to find available TPTP version
find_tptp_version() {
    local base_url="https://tptp.org/TPTP/Distribution"
    
    # Fetch directory listing and extract TPTP versions
    local version=$(curl -s "$base_url/" 2>/dev/null | \
                   grep -o 'href="TPTP-v[0-9]\+\.[0-9]\+\.[0-9]\+\.tgz"' | \
                   sed 's/href="TPTP-//;s/\.tgz"//' | \
                   sort -V | \
                   tail -1)
    
    if [ -n "$version" ]; then
        echo "$version"
    else
        # Fallback if parsing fails
        echo "v9.0.0"
    fi
}

# Check for existing TPTP installation
check_existing_tptp() {
    local tptp_path="$1"
    
    # Check for TPTP-v* directories
    for dir in "$tptp_path"/TPTP-v*; do
        if [ -d "$dir" ] && [ -d "$dir/Problems" ]; then
            local version=$(basename "$dir" | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+')
            if [ -n "$version" ]; then
                echo "$version"
                return
            fi
        fi
    done
    
    echo "none"
}

if ask_yes_no "Would you like to download and set up the TPTP library?" "n"; then
    echo ""
    
    # Expand TPTP_PATH if it contains ~
    TPTP_PATH_EXPANDED="${TPTP_PATH/#\~/$HOME}"
    
    # Check for existing installation
    existing_version=$(check_existing_tptp "$TPTP_PATH_EXPANDED")
    
    if [ "$existing_version" != "none" ]; then
        echo "Found existing TPTP installation: version $existing_version"
    fi
    
    # Find available version
    echo "Searching for available TPTP version..."
    latest_version=$(find_tptp_version)
    latest_version_num=$(echo "$latest_version" | sed 's/v//')
    
    echo "Found TPTP version: $latest_version"
    
    # Compare versions if existing installation found
    if [ "$existing_version" != "none" ] && [ "$existing_version" != "unknown" ]; then
        if [ "$existing_version" = "$latest_version_num" ]; then
            echo "You already have the latest version installed."
            if ! ask_yes_no "Would you like to reinstall?" "n"; then
                echo "Keeping existing installation."
                cd "$SCRIPT_DIR"
            else
                existing_version="none"  # Force reinstall
            fi
        else
            echo "Your version ($existing_version) differs from the available version ($latest_version_num)."
            if ! ask_yes_no "Would you like to install version $latest_version_num?" "y"; then
                echo "Keeping existing installation."
                cd "$SCRIPT_DIR"
            else
                existing_version="none"  # Force upgrade
            fi
        fi
    fi
    
    # Download and install if needed
    if [ "$existing_version" = "none" ] || [ "$existing_version" = "unknown" ]; then
        echo "TPTP library not found."
        
        # Create TPTP directory
        mkdir -p "$TPTP_PATH_EXPANDED"
        
        TPTP_ARCHIVE="$TPTP_PATH_EXPANDED/TPTP-${latest_version}.tgz"
        
        # Check if archive already exists
        if [ -f "$TPTP_ARCHIVE" ]; then
            echo "Found TPTP archive at: $TPTP_ARCHIVE"
            echo "Extracting TPTP archive..."
            cd "$TPTP_PATH_EXPANDED"
            tar -xzf "TPTP-${latest_version}.tgz"
            
            # Verify extraction
            if [ -d "TPTP-${latest_version}/Problems" ]; then
                echo "TPTP library ${latest_version} extracted successfully!"
                echo "TPTP root: $TPTP_PATH_EXPANDED/TPTP-${latest_version}"
                echo "Problems directory: $TPTP_PATH_EXPANDED/TPTP-${latest_version}/Problems"
                echo "Axioms directory: $TPTP_PATH_EXPANDED/TPTP-${latest_version}/Axioms"
            else
                echo "Warning: Problems directory not found in TPTP archive."
            fi
            
            cd "$SCRIPT_DIR"
            
            # Optionally clean up archive
            if ask_yes_no "Would you like to keep the downloaded archive?" "n"; then
                echo "Keeping archive at: $TPTP_ARCHIVE"
            else
                rm -f "$TPTP_ARCHIVE"
                echo "Archive removed."
            fi
        else
            echo ""
            echo "Downloading TPTP library..."
            TPTP_URL="https://tptp.org/TPTP/Distribution/TPTP-${latest_version}.tgz"
            
            echo "Downloading from $TPTP_URL..."
            if curl -L -o "$TPTP_ARCHIVE" "$TPTP_URL" || wget -O "$TPTP_ARCHIVE" "$TPTP_URL"; then
                echo "Download complete. Extracting TPTP archive..."
                cd "$TPTP_PATH_EXPANDED"
                tar -xzf "TPTP-${latest_version}.tgz"
                
                # Verify extraction
                if [ -d "TPTP-${latest_version}/Problems" ]; then
                    echo "TPTP library ${latest_version} extracted successfully!"
                    echo "TPTP root: $TPTP_PATH_EXPANDED/TPTP-${latest_version}"
                    echo "Problems directory: $TPTP_PATH_EXPANDED/TPTP-${latest_version}/Problems"
                    echo "Axioms directory: $TPTP_PATH_EXPANDED/TPTP-${latest_version}/Axioms"
                else
                    echo "Warning: Problems directory not found in TPTP archive."
                fi
                
                cd "$SCRIPT_DIR"
                
                # Optionally clean up archive
                if ask_yes_no "Would you like to keep the downloaded archive?" "n"; then
                    echo "Keeping archive at: $TPTP_ARCHIVE"
                else
                    rm -f "$TPTP_ARCHIVE"
                    echo "Archive removed."
                fi
            else
                echo "Error: Failed to download TPTP archive."
                echo "Please download manually from: $TPTP_URL"
                echo "and place it at: $TPTP_ARCHIVE"
                echo "Then run this setup script again."
            fi
        fi
    fi
else
    echo "Skipping TPTP library download."
    echo "You can download it manually later from: https://tptp.org/"
fi

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f .pre-commit-config.yaml ]; then
    echo ""
    echo "Installing pre-commit hooks..."
    if ! command -v pre-commit &> /dev/null; then
        conda install -y pre-commit -c conda-forge
    fi
    pre-commit install
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "ProofAtlas has been successfully installed!"
echo ""
echo "Installed components:"
echo "  ✓ Core theorem proving functionality"
echo "  ✓ TPTP parser and problem format support"
echo "  ✓ Saturation loop with given clause algorithm"
echo "  ✓ Inference rules (resolution, factoring, subsumption)"
echo "  ✓ Clause selection strategies (FIFO, LIFO, Random, Shortest, GNN)"
echo "  ✓ PyTorch and Graph Neural Network support"
if [ "$GPU_SUPPORT" = "yes" ]; then
    echo "  ✓ CUDA $CUDA_VERSION toolkit for GPU acceleration"
else
    echo "  ✓ CPU-only PyTorch (no GPU acceleration)"
fi
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify the installation, run:"
echo "  cd python && python -m pytest tests/core/test_logic.py -v"
echo ""

# Display current configuration
if [ -f .env ]; then
    echo "Current configuration (.env):"
    echo "============================="
    echo "Directory structure:"
    echo "  Data:      $DATA_DIR"
    echo "  Problems:  $PROBLEMS_DIR"
    echo "  Proofs:    $PROOFS_DIR"
    echo "  Datasets:  $DATASETS_DIR"
    echo "  Cache:     $CACHE_DIR"
    echo "  TPTP:      $TPTP_PATH"
    echo "  Logs:      $LOG_DIR"
    if [ -d "$SELECTORS_DIR" ]; then
        echo "  Selectors: $SELECTORS_DIR"
    fi
    echo ""
fi

echo "Next steps:"
echo "  1. Activate the environment: conda activate ${ENV_NAME}"
echo "  2. Run tests: cd python && python -m pytest tests/ -v"
echo "  3. Build Rust components (optional): cd rust && maturin develop"
echo "  4. Start developing with ProofAtlas!"