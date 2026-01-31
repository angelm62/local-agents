#!/bin/bash

# AI Agents Workspace - System Verification Script
# Checks prerequisites before installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track warnings and errors
WARNINGS=0
ERRORS=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Agents Workspace - System Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" == "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" == "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
        ((WARNINGS++))
    elif [ "$status" == "ERROR" ]; then
        echo -e "${RED}✗${NC} $message"
        ((ERRORS++))
    else
        echo -e "${BLUE}ℹ${NC} $message"
    fi
}

# Check Python version
echo -e "${BLUE}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    print_status "INFO" "Found Python $PYTHON_VERSION"

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ] && [ "$PYTHON_MINOR" -le 13 ]; then
        print_status "OK" "Python version is compatible (3.10-3.13)"
    elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 9 ]; then
        print_status "WARN" "Python 3.9 detected. Recommended: 3.10-3.13"
    else
        print_status "ERROR" "Python version not compatible. Required: 3.10-3.13"
    fi
else
    print_status "ERROR" "Python3 not found. Please install Python 3.10-3.13"
fi

# Check pip
echo ""
echo -e "${BLUE}Checking pip installation...${NC}"
if python3 -m pip --version &> /dev/null; then
    PIP_VERSION=$(python3 -m pip --version | awk '{print $2}')
    print_status "OK" "pip $PIP_VERSION is installed"
else
    print_status "ERROR" "pip not found. Please install pip"
fi

# Check for CUDA/NVIDIA drivers
echo ""
echo -e "${BLUE}Checking GPU and CUDA installation...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    print_status "OK" "NVIDIA GPU detected: $GPU_INFO"

    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "Unknown")
    if [ "$CUDA_VERSION" != "Unknown" ]; then
        print_status "OK" "CUDA Version: $CUDA_VERSION"
    else
        print_status "WARN" "Could not detect CUDA version"
    fi

    # Check GPU utilization
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
    print_status "INFO" "Current GPU utilization: ${GPU_UTIL}%"

    # Check GPU memory
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
    GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    print_status "INFO" "GPU Memory: ${GPU_MEM_USED}MB / ${GPU_MEM_TOTAL}MB used (${GPU_MEM_FREE}MB free)"

    if [ "$GPU_MEM_FREE" -lt 2048 ]; then
        print_status "WARN" "Less than 2GB GPU memory free. May need to free up VRAM"
    fi
else
    print_status "WARN" "nvidia-smi not found. GPU acceleration may not be available"
fi

# Check disk space
echo ""
echo -e "${BLUE}Checking disk space...${NC}"
DISK_AVAIL=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
print_status "INFO" "Available disk space: ${DISK_AVAIL}GB"

if [ "$DISK_AVAIL" -lt 10 ]; then
    print_status "ERROR" "Less than 10GB disk space available. Installation requires at least 10GB"
elif [ "$DISK_AVAIL" -lt 20 ]; then
    print_status "WARN" "Less than 20GB disk space available. Recommended: 20GB+"
else
    print_status "OK" "Sufficient disk space available"
fi

# Check internet connectivity
echo ""
echo -e "${BLUE}Checking internet connectivity...${NC}"
if ping -c 1 8.8.8.8 &> /dev/null; then
    print_status "OK" "Internet connection available"
else
    print_status "WARN" "No internet connection detected. Installation may fail"
fi

# Check Ollama (optional)
echo ""
echo -e "${BLUE}Checking Ollama installation (optional)...${NC}"
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version | head -n1 || echo "Unknown")
    print_status "OK" "Ollama installed: $OLLAMA_VERSION"

    # Try to connect to Ollama
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_status "OK" "Ollama server is running"
        MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || echo "")
        if [ -n "$MODELS" ]; then
            print_status "INFO" "Available Ollama models: $(echo $MODELS | tr '\n' ', ' | sed 's/,$//')"
        fi
    else
        print_status "WARN" "Ollama installed but server not running. Start with: ollama serve"
    fi
else
    print_status "INFO" "Ollama not installed (optional)"
fi

# Check virtual environment support
echo ""
echo -e "${BLUE}Checking virtual environment support...${NC}"
if python3 -m venv --help &> /dev/null; then
    print_status "OK" "venv module available"
else
    print_status "ERROR" "venv module not available. Install with: apt install python3-venv"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Verification Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! System is ready for installation.${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Verification completed with $WARNINGS warning(s).${NC}"
    echo -e "${YELLOW}Installation can proceed, but some features may not work optimally.${NC}"
    exit 0
else
    echo -e "${RED}Verification failed with $ERRORS error(s) and $WARNINGS warning(s).${NC}"
    echo -e "${RED}Please resolve the errors before proceeding with installation.${NC}"
    exit 1
fi
