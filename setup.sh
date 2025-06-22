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

# Optional installations
echo ""
echo "=== Optional Components ==="
echo ""
echo "ProofAtlas can work with additional optional components:"
echo ""

# PyTorch and Machine Learning packages
echo "1. PyTorch and Graph Neural Networks"
echo "   - Required for: GNN-based clause selection, learned proof guidance"
echo "   - Packages: pytorch, pytorch-cuda, pyg (PyTorch Geometric), pytorch-lightning"
echo "   - Note: This requires a compatible NVIDIA GPU for CUDA support"
echo ""

if ask_yes_no "Would you like to install PyTorch and GNN packages?"; then
    echo "Installing PyTorch ecosystem..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    
    echo "Installing PyTorch Geometric..."
    conda install -y pyg -c pyg
    
    echo "Installing PyTorch Lightning..."
    conda install -y pytorch-lightning torchmetrics -c conda-forge
    
    echo "PyTorch and GNN packages installed successfully!"
else
    echo "Skipping PyTorch installation."
    echo "Note: GNN-based clause selection will not be available."
fi

echo ""

# Claude CLI
echo "2. Claude CLI (AI Assistant)"
echo "   - Required for: Interactive AI assistance with code and theorem proving"
echo "   - Packages: nodejs (via conda), claude (via npm)"
echo "   - Note: Requires a Claude API key"
echo ""

if ask_yes_no "Would you like to install Claude CLI?"; then
    echo "Installing Node.js via conda..."
    conda install -y nodejs -c conda-forge
    
    echo "Installing Claude CLI in the conda environment..."
    # Install locally to the conda environment, not globally
    npm install @anthropic-ai/claude-cli
    
    # Add node_modules/.bin to PATH for this environment
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export PATH="$CONDA_PREFIX/node_modules/.bin:$PATH"' > $CONDA_PREFIX/etc/conda/activate.d/npm.sh
    chmod +x $CONDA_PREFIX/etc/conda/activate.d/npm.sh
    
    # Also create deactivate script
    mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
    echo 'export PATH="${PATH//$CONDA_PREFIX\/node_modules\/.bin:/}"' > $CONDA_PREFIX/etc/conda/deactivate.d/npm.sh
    chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/npm.sh
    
    echo "Claude CLI installed successfully!"
    echo ""
    echo "To use Claude CLI, you'll need to:"
    echo "  1. Reactivate the conda environment: conda deactivate && conda activate ${ENV_NAME}"
    echo "  2. Set up your API key: export ANTHROPIC_API_KEY='your-api-key-here'"
    echo "  3. Or run: claude login"
else
    echo "Skipping Claude CLI installation."
fi

# Install the package in development mode
echo ""
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

# Check if .env exists and ask about overwriting
if [ -f .env ]; then
    echo "An .env file already exists."
    if ask_yes_no "Do you want to update it?" "n"; then
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
    else
        echo "Keeping existing .env file."
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

# Function to get latest TPTP version
get_latest_tptp_version() {
    echo "Checking for latest TPTP version..."
    
    # Try to fetch the TPTP distribution page and extract version numbers
    local versions=$(curl -s "https://www.tptp.org/TPTP/Distribution/" | grep -oE 'TPTP-v[0-9]+\.[0-9]+\.[0-9]+\.tgz' | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -1)
    
    if [ -z "$versions" ]; then
        # Fallback: try alternative URL or use a known recent version
        versions=$(curl -s "http://www.tptp.org/TPTP/Distribution/" | grep -oE 'TPTP-v[0-9]+\.[0-9]+\.[0-9]+\.tgz' | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -1)
    fi
    
    if [ -z "$versions" ]; then
        echo "Warning: Could not determine latest TPTP version automatically."
        echo "Using fallback version v8.2.0"
        echo "v8.2.0"
    else
        echo "$versions"
    fi
}

# Check for existing TPTP installation
check_existing_tptp() {
    local tptp_path="$1"
    
    if [ -d "$tptp_path/Problems" ] || [ -L "$tptp_path/Problems" ]; then
        # Check if it's a symlink and follow it
        if [ -L "$tptp_path/Problems" ]; then
            local target=$(readlink "$tptp_path/Problems")
            if [[ "$target" =~ TPTP-v([0-9]+\.[0-9]+\.[0-9]+) ]]; then
                echo "${BASH_REMATCH[1]}"
                return
            fi
        fi
        
        # Check for version file or directory names
        for dir in "$tptp_path"/TPTP-v*; do
            if [ -d "$dir" ]; then
                local version=$(basename "$dir" | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | sed 's/v//')
                if [ -n "$version" ]; then
                    echo "$version"
                    return
                fi
            fi
        done
        
        echo "unknown"
    else
        echo "none"
    fi
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
    
    # Get latest version
    latest_version=$(get_latest_tptp_version)
    latest_version_num=$(echo "$latest_version" | sed 's/v//')
    
    echo "Latest TPTP version available: $latest_version"
    
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
            echo "Your version ($existing_version) is older than the latest version ($latest_version_num)."
            if ! ask_yes_no "Would you like to upgrade?" "y"; then
                echo "Keeping existing installation."
                cd "$SCRIPT_DIR"
            else
                existing_version="none"  # Force upgrade
            fi
        fi
    fi
    
    # Download and install if needed
    if [ "$existing_version" = "none" ] || [ "$existing_version" = "unknown" ]; then
        echo "Downloading TPTP library..."
        
        # Create TPTP directory
        mkdir -p "$TPTP_PATH_EXPANDED"
        
        # Download TPTP archive
        TPTP_URL="http://www.tptp.org/TPTP/Distribution/TPTP-${latest_version}.tgz"
        TPTP_ARCHIVE="$TPTP_PATH_EXPANDED/TPTP-${latest_version}.tgz"
        
        if [ -f "$TPTP_ARCHIVE" ]; then
            echo "TPTP archive already exists. Using existing archive."
        else
            echo "Downloading from $TPTP_URL..."
            if ! curl -L -o "$TPTP_ARCHIVE" "$TPTP_URL" && ! wget -O "$TPTP_ARCHIVE" "$TPTP_URL"; then
                echo "Error: Failed to download TPTP archive."
                echo "Please check your internet connection or download manually from:"
                echo "  $TPTP_URL"
                cd "$SCRIPT_DIR"
            else
                # Extract TPTP archive
                echo "Extracting TPTP archive..."
                cd "$TPTP_PATH_EXPANDED"
                tar -xzf "TPTP-${latest_version}.tgz"
                
                # Create symlink to Problems directory
                if [ -d "TPTP-${latest_version}/Problems" ]; then
                    # Remove old symlink if exists
                    [ -L "Problems" ] && rm -f "Problems"
                    ln -sf "TPTP-${latest_version}/Problems" "Problems"
                    echo "TPTP library ${latest_version} extracted successfully!"
                    echo "Problems directory: $TPTP_PATH_EXPANDED/Problems"
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
            fi
        fi
    fi
else
    echo "Skipping TPTP library download."
    echo "You can download it manually later from: http://www.tptp.org/"
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
echo "Core ProofAtlas components have been installed."
echo ""
echo "Installed components:"
echo "  ✓ Core theorem proving functionality"
echo "  ✓ TPTP and Vampire parsers"
echo "  ✓ Saturation loop and inference rules"
echo "  ✓ Basic clause selection (FIFO, Random)"

# Check what optional components were installed
echo ""
echo "Optional components:"
if python -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch and GNN support"
else
    echo "  ✗ PyTorch and GNN support (not installed)"
fi

if [ -f "$CONDA_PREFIX/node_modules/.bin/claude" ]; then
    echo "  ✓ Claude CLI"
else
    echo "  ✗ Claude CLI (not installed)"
fi

echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify the installation, run:"
echo "  python -m pytest tests/core/test_logic.py -v"
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
echo "  2. Run tests: python -m pytest tests/ -v"
echo "  3. Explore examples: python examples/basic_saturation.py"
echo "  4. Start developing with ProofAtlas!"