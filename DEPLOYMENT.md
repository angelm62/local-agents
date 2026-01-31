# Deployment Guide

Quick reference for deploying the AI Agents Workspace to your Ubuntu Server.

## Pre-Deployment Checklist

### Local Machine (Mac/Development)

- [x] All files created and committed to Git
- [ ] Pushed to Git server (GitHub, GitLab, or private Git server)
- [ ] Confirmed repository URL

### Ubuntu Server (10.27.27.190)

- [ ] SSH access working: `ssh user@10.27.27.190`
- [ ] Python 3.10+ installed: `python3 --version`
- [ ] NVIDIA drivers installed: `nvidia-smi`
- [ ] Git installed: `git --version`
- [ ] 20GB+ free disk space: `df -h`

## Deployment Steps

### 1. Push to Git Server

From your local machine:

```bash
cd ~/ai-agents-workspace

# Add your Git remote (choose one):

# GitHub
git remote add origin https://github.com/yourusername/ai-agents-workspace.git

# GitLab
git remote add origin https://gitlab.com/yourusername/ai-agents-workspace.git

# Private Git server
git remote add origin git@your-server.com:path/to/repo.git

# Push to remote
git push -u origin main
```

### 2. SSH to Ubuntu Server

```bash
ssh user@10.27.27.190
```

### 3. Clone Repository

```bash
cd ~
git clone <your-repo-url> ai-agents-workspace
cd ai-agents-workspace
```

### 4. Verify System Requirements

```bash
./setup/verify_setup.sh
```

This checks:
- Python version (3.10-3.13)
- GPU and CUDA availability
- Disk space
- Ollama (optional)

Expected output: All checks should pass or show warnings only.

### 5. Run Installation

```bash
./setup/install.sh
```

Installation process:
1. Creates Python virtual environment
2. Auto-detects CUDA version
3. Installs PyTorch with CUDA support
4. Installs LangChain and CrewAI
5. Installs all dependencies
6. Verifies installation

This takes 5-15 minutes depending on internet speed.

### 6. Configure Environment

```bash
cp setup/.env.template .env
nano .env
```

Update as needed:
- `HUGGINGFACE_TOKEN` - Only if using gated models
- `OLLAMA_BASE_URL` - Default is fine if Ollama runs locally
- `CUDA_VISIBLE_DEVICES` - GPU selection (default: 0)

### 7. Activate Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` in your prompt.

### 8. Verify Installation

```bash
# Check GPU
python shared/gpu_monitor.py

# Check system info
python shared/utils.py

# Verify imports
python -c "import torch, langchain, crewai; print('All packages imported successfully')"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Testing

### Quick Tests

```bash
# Activate environment if not already active
source venv/bin/activate

# Test GPU monitoring
python shared/gpu_monitor.py

# Test Ollama connection (if installed)
curl http://localhost:11434/api/tags
```

### LangChain Tests

```bash
# Basic agent with Ollama (requires Ollama running)
python langchain-examples/basic_agent.py

# GPU verification with HuggingFace
python langchain-examples/gpu_verification.py

# Local HuggingFace models
python langchain-examples/huggingface_local.py

# Batch processing
python langchain-examples/batch_processing.py
```

### CrewAI Tests

```bash
# Basic crew (requires Ollama)
python crewai-examples/basic_crew.py

# Multi-agent collaboration
python crewai-examples/multi_agent_task.py

# Tools demonstration
python crewai-examples/tools_demo.py

# HuggingFace powered crew
python crewai-examples/huggingface_crew.py
```

## Ollama Setup (If Not Installed)

If you don't have Ollama installed:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# Start server (in background)
ollama serve &

# Verify
ollama list
```

## Troubleshooting

### Installation Failed

```bash
# Check logs
cat setup/install.log

# Look for errors
grep -i error setup/install.log

# Re-run installation
./setup/install.sh
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA in PyTorch
source venv/bin/activate
python -c "import torch; print(torch.version.cuda)"
```

### Ollama Not Working

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check installed models
ollama list
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean pip cache
pip cache purge

# Clean HuggingFace cache (careful!)
rm -rf ~/.cache/huggingface/hub/*
```

## Post-Deployment

### Update Repository

To pull updates:

```bash
cd ~/ai-agents-workspace
git pull origin main
source venv/bin/activate
pip install -r setup/requirements.txt --upgrade
```

### Start Development

```bash
# Always activate environment first
cd ~/ai-agents-workspace
source venv/bin/activate

# Create new examples in langchain-examples/ or crewai-examples/
# Use shared utilities from shared/
```

### Production Use

For production workloads:

1. **Systemd Service**: Create service to auto-start Ollama
2. **GPU Monitoring**: Set up continuous monitoring
3. **Model Caching**: Pre-download models to cache
4. **Resource Limits**: Configure GPU memory limits
5. **Logging**: Set up centralized logging

## Quick Reference

### Common Commands

```bash
# Activate environment
source venv/bin/activate

# Check GPU
python shared/gpu_monitor.py

# Monitor GPU continuously
python shared/gpu_monitor.py monitor

# Run examples
python langchain-examples/basic_agent.py
python crewai-examples/basic_crew.py

# Update repository
git pull origin main
pip install -r setup/requirements.txt --upgrade
```

### File Locations

- **Installation log**: `setup/install.log`
- **Environment config**: `.env`
- **Model cache**: `~/.cache/huggingface/`
- **Virtual environment**: `venv/`

### Important Paths

```bash
# Project root
cd ~/ai-agents-workspace

# Activate environment
source ~/ai-agents-workspace/venv/bin/activate

# Configuration
nano ~/ai-agents-workspace/.env
```

## Support

Issues? Check:

1. `setup/install.log` for installation errors
2. `README.md` Troubleshooting section
3. Run `./setup/verify_setup.sh` again
4. Check GPU with `nvidia-smi`
5. Verify Ollama with `ollama list`

## Success Criteria

Deployment is successful when:

- [x] Git repository cloned
- [x] Verification script passes
- [x] Installation completes without errors
- [x] Virtual environment activates
- [x] GPU detected by PyTorch
- [x] At least one LangChain example runs
- [x] At least one CrewAI example runs

---

**Target Server**: Ubuntu Server 22.04 @ 10.27.27.190
**GPU**: NVIDIA RTX 5060 Ti
**CUDA**: 12.1+
**Python**: 3.10+
