<<<<<<< HEAD
# local-agents
=======
# AI Agents Workspace

A comprehensive development environment for building AI agents using LangChain and CrewAI frameworks with GPU acceleration support.

## Overview

This workspace provides a complete setup for developing AI agents on Ubuntu Server with:

- **LangChain**: Framework for building applications with LLMs
- **CrewAI**: Multi-agent orchestration framework
- **GPU Acceleration**: CUDA support for HuggingFace models
- **Local Models**: Ollama integration + HuggingFace models
- **Ready-to-Run Examples**: Practical demonstrations for both frameworks

## Prerequisites

### System Requirements

- **OS**: Ubuntu Server 22.04 (or compatible Linux distribution)
- **Python**: 3.10 - 3.13
- **GPU**: NVIDIA GPU with CUDA drivers (optional but recommended)
- **Disk Space**: 20GB+ free space
- **RAM**: 8GB minimum (16GB+ recommended)

### Recommended Hardware

- NVIDIA RTX GPU (RTX 3060+, RTX 4060+, RTX 5060+ or better)
- CUDA 11.7+ drivers installed
- 16GB+ system RAM
- SSD storage for model caching

### Software Dependencies

- Python 3.10+ with pip
- nvidia-smi (for GPU monitoring)
- Git (for repository management)
- Ollama (optional, for local Ollama models)

## Installation

### Quick Start (Ubuntu Server)

1. **Clone the repository**

```bash
git clone <your-repo-url> ~/ai-agents-workspace
cd ~/ai-agents-workspace
```

2. **Run system verification**

```bash
chmod +x setup/verify_setup.sh
./setup/verify_setup.sh
```

This checks your Python version, GPU availability, disk space, and dependencies.

3. **Run installation**

```bash
chmod +x setup/install.sh
./setup/install.sh
```

The installation script will:
- Create a Python virtual environment
- Auto-detect your CUDA version
- Install PyTorch with matching CUDA support
- Install LangChain, CrewAI, and all dependencies
- Configure the environment
- Verify the installation

Installation takes 5-15 minutes depending on your internet connection.

4. **Activate the environment**

```bash
source venv/bin/activate
```

### Manual Installation

If you prefer to install manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r setup/requirements.txt

# Create .env file
cp setup/.env.template .env
```

## Configuration

### Environment Variables

Copy the template and configure:

```bash
cp setup/.env.template .env
nano .env
```

Key configuration options:

```bash
# HuggingFace Token (optional, for gated models)
HUGGINGFACE_TOKEN=your_token_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# GPU Selection (0 for first GPU)
CUDA_VISIBLE_DEVICES=0

# Model Cache Location (optional)
TRANSFORMERS_CACHE=/path/to/model/cache
```

### HuggingFace Token

Only required for gated models (like Llama). Get your token:
1. Visit https://huggingface.co/settings/tokens
2. Create a new token with read access
3. Add to `.env` file

### Ollama Setup (Optional)

If you want to use Ollama models:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# Start server (if not running)
ollama serve
```

## Project Structure

```
ai-agents-workspace/
├── setup/                      # Installation and configuration
│   ├── install.sh             # Main installation script
│   ├── verify_setup.sh        # System verification
│   ├── requirements.txt       # Python dependencies
│   └── .env.template          # Environment template
│
├── langchain-examples/        # LangChain demonstrations
│   ├── basic_agent.py         # Simple agent with Ollama
│   ├── batch_processing.py    # Async batch processing
│   ├── gpu_verification.py    # GPU usage verification
│   └── huggingface_local.py   # Local HF models + embeddings
│
├── crewai-examples/           # CrewAI demonstrations
│   ├── basic_crew.py          # Single agent crew
│   ├── multi_agent_task.py    # Multi-agent collaboration
│   ├── tools_demo.py          # Custom tools usage
│   └── huggingface_crew.py    # Crew with HF models
│
├── shared/                    # Common utilities
│   ├── utils.py               # Helper functions
│   └── gpu_monitor.py         # GPU monitoring tools
│
├── venv/                      # Virtual environment (created during install)
├── .env                       # Your configuration (create from template)
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Usage Examples

### Verify Installation

```bash
# Activate environment
source venv/bin/activate

# Check GPU status
python shared/gpu_monitor.py

# Print system info
python shared/utils.py

# Monitor GPU in real-time
python shared/gpu_monitor.py monitor
```

### LangChain Examples

#### 1. Basic Agent (Ollama)

```bash
python langchain-examples/basic_agent.py
```

Creates a simple conversational agent using Ollama. Tests basic question-answering capabilities.

#### 2. Batch Processing

```bash
python langchain-examples/batch_processing.py
```

Demonstrates asynchronous batch processing of multiple inputs for efficient parallel execution.

#### 3. GPU Verification

```bash
python langchain-examples/gpu_verification.py
```

Loads a HuggingFace model on GPU, monitors VRAM usage, and verifies GPU utilization during inference.

#### 4. HuggingFace Local Models

```bash
python langchain-examples/huggingface_local.py
```

Uses local HuggingFace models (TinyLlama) with GPU acceleration. Includes embeddings demonstration.

### CrewAI Examples

#### 1. Basic Crew

```bash
python crewai-examples/basic_crew.py
```

Simple crew with a research agent using Ollama backend.

#### 2. Multi-Agent Task

```bash
python crewai-examples/multi_agent_task.py
```

Three agents collaborate sequentially:
- Researcher gathers information
- Analyst extracts insights
- Writer creates an article

#### 3. Tools Demo

```bash
python crewai-examples/tools_demo.py
```

Demonstrates agents using custom Python functions as tools (calculator, time checker, word counter).

#### 4. HuggingFace Crew

```bash
python crewai-examples/huggingface_crew.py
```

Crew powered by local HuggingFace models with GPU acceleration.

## Architecture

### LangChain

LangChain provides components for building LLM applications:

- **LLMs**: Interface to language models (Ollama, HuggingFace)
- **Prompts**: Template management for model inputs
- **Chains**: Combine LLMs with prompts for complex workflows
- **Embeddings**: Vector representations for semantic search
- **Memory**: Conversation history management

### CrewAI

CrewAI enables multi-agent orchestration:

- **Agents**: Autonomous entities with roles and goals
- **Tasks**: Specific objectives assigned to agents
- **Crews**: Groups of agents working together
- **Tools**: Functions agents can use to perform actions
- **Processes**: Sequential or hierarchical workflows

### Integration Points

Both frameworks integrate seamlessly:

- **Ollama**: Local model inference via REST API
- **HuggingFace**: Direct model loading with GPU support
- **Tools**: Shared tool definitions across frameworks
- **LLM Backend**: Interchangeable model backends

## GPU Acceleration

### Supported Models

The workspace supports GPU acceleration for:

- **HuggingFace Models**: transformers library with CUDA
- **PyTorch Models**: Native GPU tensor operations
- **Quantized Models**: 8-bit and 4-bit quantization via bitsandbytes

### Recommended Models

For testing and development:

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| TinyLlama-1.1B | 1.1GB | 2-3GB | Testing, development |
| Phi-2 | 2.7GB | 4-6GB | Good quality, reasonable size |
| Mistral-7B | 7B | 8-10GB | Production use |
| Llama-2-13B | 13B | 14-16GB | High quality responses |

### GPU Monitoring

Monitor GPU usage during inference:

```bash
# One-time snapshot
python shared/gpu_monitor.py

# Continuous monitoring (2-second intervals)
python shared/gpu_monitor.py monitor

# Custom interval (5 seconds)
python shared/gpu_monitor.py monitor 5

# Monitor for specific duration (60 seconds)
python shared/gpu_monitor.py monitor 2 60
```

### Memory Management

Tips for managing GPU memory:

```python
# Use smaller models for testing
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Use 8-bit quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8, 0)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:
```bash
# Check nvidia-smi
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Check CUDA environment
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

#### 2. Ollama Connection Failed

**Symptoms**: "Ollama server not available"

**Solutions**:
```bash
# Start Ollama server
ollama serve

# Check if running
curl http://localhost:11434/api/tags

# Install a model
ollama pull llama2
ollama list
```

#### 3. Out of Memory (OOM)

**Symptoms**: CUDA OOM error during model loading

**Solutions**:
```bash
# Use smaller model
# In code: model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Use quantization
# In code: use_8bit=True or use_4bit=True

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check GPU usage
python shared/gpu_monitor.py
```

#### 4. Model Download Slow/Failed

**Symptoms**: HuggingFace model download timeout

**Solutions**:
```bash
# Set custom cache location with more space
export TRANSFORMERS_CACHE=/path/to/larger/disk

# Download separately
python -c "from transformers import AutoModel; AutoModel.from_pretrained('model-name')"

# Check disk space
df -h
```

#### 5. Import Errors

**Symptoms**: `ImportError: No module named 'xxx'`

**Solutions**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r setup/requirements.txt

# Check installation
pip list | grep -E "torch|langchain|crewai"
```

### Logs and Debugging

Installation logs are saved to `setup/install.log`:

```bash
# View installation log
cat setup/install.log

# Check for errors
grep -i error setup/install.log
```

Enable verbose output in examples:

```python
# LangChain
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# CrewAI
agent = Agent(..., verbose=True)
crew = Crew(..., verbose=2)
```

## Performance Tips

### Optimize Inference Speed

1. **Use GPU**: Always use GPU for HuggingFace models
2. **Quantization**: Use 8-bit or 4-bit for faster inference
3. **Smaller Models**: Start with TinyLlama, scale up as needed
4. **Batch Processing**: Process multiple inputs together
5. **Caching**: Enable model caching to avoid reloading

### Memory Optimization

```python
# Use quantization
from shared.utils import load_huggingface_model

model, tokenizer = load_huggingface_model(
    "model-name",
    use_4bit=True  # Reduces memory by ~4x
)

# Use half precision
model, tokenizer = load_huggingface_model(
    "model-name",
    device="cuda",
    torch_dtype=torch.float16
)
```

### Production Deployment

For production workloads:

1. **Use larger models**: Mistral-7B or Llama-2-13B
2. **Load balancing**: Multiple Ollama instances
3. **Model caching**: Pre-download models
4. **GPU pooling**: Multiple GPUs with device mapping
5. **Monitoring**: Track GPU utilization and response times

## Development

### Adding New Examples

Create new examples in the appropriate directory:

```python
# langchain-examples/my_example.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.utils import check_gpu_available
# Your code here
```

### Using Custom Models

Swap models easily:

```python
# Use different Ollama model
llm = Ollama(model="mistral")

# Use different HuggingFace model
model_name = "microsoft/phi-2"
```

### Creating Custom Tools

Define tools for CrewAI agents:

```python
from crewai_tools import tool

@tool("MyCustomTool")
def my_tool(input: str) -> str:
    """Tool description for the agent"""
    # Your logic here
    return result
```

## Next Steps

### Learning Resources

- **LangChain Docs**: https://python.langchain.com/docs/
- **CrewAI Docs**: https://docs.crewai.com/
- **HuggingFace Docs**: https://huggingface.co/docs
- **PyTorch Tutorials**: https://pytorch.org/tutorials/

### Extension Ideas

1. **RAG System**: Add document retrieval with vector databases
2. **Web Scraping**: Integrate web scraping tools
3. **API Integration**: Connect to external APIs
4. **Memory Systems**: Add persistent conversation memory
5. **Fine-tuning**: Fine-tune models on custom data
6. **Model Serving**: Deploy as REST API with FastAPI
7. **Monitoring**: Add observability with LangSmith or Weights & Biases

### Advanced Topics

- **LangGraph**: Build stateful multi-agent systems
- **Vector Stores**: FAISS, ChromaDB, Pinecone integration
- **Function Calling**: Structured outputs and tool use
- **Streaming**: Real-time token streaming
- **Evaluation**: LLM output evaluation and metrics

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review example code for usage patterns
3. Check logs in `setup/install.log`
4. Open an issue with details about your environment

## Acknowledgments

- **LangChain**: Framework for LLM applications
- **CrewAI**: Multi-agent orchestration
- **HuggingFace**: Model hub and transformers
- **Ollama**: Local model inference
- **PyTorch**: Deep learning framework

---

**Built for Ubuntu Server 22.04 with NVIDIA RTX 5060 Ti**

Happy agent building!
>>>>>>> 5c26794 (Initial commit: AI Agents Workspace setup)
