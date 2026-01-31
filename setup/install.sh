#!/bin/bash

# AI Agents Workspace - Automated Installation Script
# Fully automated setup with no user prompts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/install.log"

# Redirect all output to log file as well
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Agents Workspace - Installation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Installation log: $LOG_FILE"
echo ""

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" == "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" == "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    elif [ "$status" == "ERROR" ]; then
        echo -e "${RED}✗${NC} $message"
    else
        echo -e "${BLUE}➜${NC} $message"
    fi
}

# Function to handle errors
handle_error() {
    echo ""
    print_status "ERROR" "Installation failed at: $1"
    echo -e "${RED}Check the log file for details: $LOG_FILE${NC}"
    exit 1
}

# Trap errors
trap 'handle_error "$BASH_COMMAND"' ERR

# Phase 1: System Verification
echo -e "${BLUE}[Phase 1/5] System Verification${NC}"
print_status "INFO" "Checking Python version..."

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    print_status "OK" "Python $PYTHON_VERSION detected"
else
    print_status "ERROR" "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

# Detect CUDA version
print_status "INFO" "Detecting CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
    if [ -n "$CUDA_VERSION" ]; then
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        print_status "OK" "CUDA $CUDA_VERSION detected"

        # Determine PyTorch CUDA variant
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            TORCH_CUDA="cu121"
        elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
            TORCH_CUDA="cu118"
        else
            TORCH_CUDA="cu117"
        fi
        print_status "INFO" "Will install PyTorch for CUDA variant: $TORCH_CUDA"
    else
        print_status "WARN" "Could not detect CUDA version, will install CPU-only PyTorch"
        TORCH_CUDA="cpu"
    fi
else
    print_status "WARN" "nvidia-smi not found, will install CPU-only PyTorch"
    TORCH_CUDA="cpu"
fi

# Check disk space
DISK_AVAIL=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$DISK_AVAIL" -lt 10 ]; then
    print_status "WARN" "Only ${DISK_AVAIL}GB disk space available (recommended: 20GB+)"
else
    print_status "OK" "${DISK_AVAIL}GB disk space available"
fi

echo ""

# Phase 2: Virtual Environment Setup
echo -e "${BLUE}[Phase 2/5] Virtual Environment Setup${NC}"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    print_status "WARN" "Virtual environment already exists, removing..."
    rm -rf venv
fi

print_status "INFO" "Creating virtual environment..."
python3 -m venv venv
print_status "OK" "Virtual environment created"

print_status "INFO" "Activating virtual environment..."
source venv/bin/activate

print_status "INFO" "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet
print_status "OK" "Build tools upgraded"

echo ""

# Phase 3: Core Dependencies
echo -e "${BLUE}[Phase 3/5] Installing Core Dependencies${NC}"

# Install PyTorch with CUDA support
print_status "INFO" "Installing PyTorch with CUDA support ($TORCH_CUDA)..."
if [ "$TORCH_CUDA" == "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_CUDA --quiet
fi
print_status "OK" "PyTorch installed"

# Install HuggingFace Stack
print_status "INFO" "Installing HuggingFace stack..."
pip install transformers accelerate bitsandbytes sentence-transformers huggingface-hub --quiet
print_status "OK" "HuggingFace stack installed"

# Install LangChain
print_status "INFO" "Installing LangChain..."
pip install langchain langchain-community langchain-huggingface langchain-core --quiet
print_status "OK" "LangChain installed"

# Install CrewAI
print_status "INFO" "Installing CrewAI..."
pip install crewai crewai-tools --quiet
print_status "OK" "CrewAI installed"

# Install supporting libraries
print_status "INFO" "Installing supporting libraries..."
pip install python-dotenv psutil nvidia-ml-py3 GPUtil requests pydantic aiohttp tenacity --quiet
print_status "OK" "Supporting libraries installed"

echo ""

# Phase 4: Configuration
echo -e "${BLUE}[Phase 4/5] Configuration${NC}"

if [ ! -f ".env" ]; then
    print_status "INFO" "Creating .env file from template..."
    cp setup/.env.template .env
    print_status "OK" ".env file created (please configure as needed)"
else
    print_status "INFO" ".env file already exists, skipping"
fi

echo ""

# Phase 5: Verification
echo -e "${BLUE}[Phase 5/5] Installation Verification${NC}"

print_status "INFO" "Verifying package installations..."

# Test imports
python3 << 'EOF'
import sys

packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'langchain': 'LangChain',
    'crewai': 'CrewAI',
    'accelerate': 'Accelerate',
    'sentence_transformers': 'Sentence Transformers'
}

failed = []
for module, name in packages.items():
    try:
        __import__(module)
    except ImportError:
        failed.append(name)

if failed:
    print(f"Failed to import: {', '.join(failed)}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_status "OK" "All packages imported successfully"
else
    print_status "ERROR" "Some packages failed to import"
    exit 1
fi

# Verify CUDA availability
print_status "INFO" "Checking CUDA availability in PyTorch..."
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" == "True" ]; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    print_status "OK" "CUDA available: $GPU_COUNT GPU(s) detected - $GPU_NAME"
else
    print_status "WARN" "CUDA not available in PyTorch (CPU-only mode)"
fi

# Check Ollama connectivity (non-blocking)
print_status "INFO" "Testing Ollama connectivity..."
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    print_status "OK" "Ollama server is accessible"
else
    print_status "WARN" "Ollama server not accessible (start with: ollama serve)"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "  2. Configure your environment (optional):"
echo "     ${YELLOW}nano .env${NC}"
echo "     - Add HuggingFace token for gated models"
echo "     - Adjust CUDA settings if needed"
echo ""
echo "  3. Test the installation:"
echo "     ${YELLOW}python shared/gpu_monitor.py${NC}"
echo "     ${YELLOW}python langchain-examples/basic_agent.py${NC}"
echo "     ${YELLOW}python crewai-examples/basic_crew.py${NC}"
echo ""
echo "  4. Explore examples:"
echo "     - LangChain: langchain-examples/"
echo "     - CrewAI: crewai-examples/"
echo ""
echo "Documentation: README.md"
echo ""
