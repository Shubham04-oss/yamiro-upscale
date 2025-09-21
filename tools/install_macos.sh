#!/bin/bash

# Yamiro Upscaler - macOS Setup Script
# Installs dependencies and sets up the environment

set -e

echo "🎌 Yamiro Upscaler - macOS Setup"
echo "================================"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check for Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "✅ Apple Silicon detected (M-series)"
else
    echo "ℹ️ Intel Mac detected"
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed"
fi

# Check for Miniforge/Conda
if ! command -v conda &> /dev/null; then
    echo "🐍 Installing Miniforge..."
    if [[ "$ARCH" == "arm64" ]]; then
        curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
        bash Miniforge3-MacOSX-arm64.sh -b -p $HOME/miniforge3
    else
        curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
        bash Miniforge3-MacOSX-x86_64.sh -b -p $HOME/miniforge3
    fi
    
    # Initialize conda
    $HOME/miniforge3/bin/conda init zsh
    $HOME/miniforge3/bin/conda init bash
    
    echo "⚠️  Please restart your terminal and run this script again"
    exit 0
else
    echo "✅ Conda already installed"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew install ffmpeg git-lfs

# Create conda environment
echo "🔧 Creating conda environment..."
if conda env list | grep -q yamiro-upscale; then
    echo "ℹ️ Environment already exists, updating..."
    conda env update -f environment.yml
else
    conda env create -f environment.yml
fi

# Fix OpenMP library conflict (common on macOS)
echo "🔧 Fixing OpenMP library conflict..."
if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.zshrc; then
    echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.zshrc
    echo "Added KMP_DUPLICATE_LIB_OK to ~/.zshrc"
fi

if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.bash_profile; then
    echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.bash_profile
    echo "Added KMP_DUPLICATE_LIB_OK to ~/.bash_profile"
fi

# Set for current session
export KMP_DUPLICATE_LIB_OK=TRUE

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate yamiro-upscale"
echo "2. Install PyTorch with MPS support:"
echo "   pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu"
echo "3. Set environment variable: export KMP_DUPLICATE_LIB_OK=TRUE"
echo "4. Test the installation: python src/cli.py info"
echo "5. Try upscaling an image: python src/cli.py upscale -i path/to/image.jpg -o results/"