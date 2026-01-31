"""
Common utilities for AI Agents Workspace
Provides helper functions for model loading, environment checks, and configuration
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_gpu_available() -> bool:
    """Check if GPU is available for PyTorch"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        print("Warning: PyTorch not installed")
        return False


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}

        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        }

        # Get memory info
        if torch.cuda.is_available():
            info["memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**3  # GB
            info["memory_reserved"] = torch.cuda.memory_reserved(0) / 1024**3  # GB
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        return info
    except Exception as e:
        return {"available": False, "error": str(e)}


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default"""
    return os.getenv(key, default)


def check_ollama_available() -> bool:
    """Check if Ollama server is accessible"""
    try:
        import requests
        response = requests.get(
            f"{get_env('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
            timeout=2
        )
        return response.status_code == 200
    except Exception:
        return False


def get_ollama_models() -> list:
    """Get list of available Ollama models"""
    try:
        import requests
        response = requests.get(
            f"{get_env('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
            timeout=2
        )
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except Exception:
        return []


def load_huggingface_model(
    model_name: str,
    device: str = "auto",
    use_8bit: bool = False,
    use_4bit: bool = False
):
    """
    Load a HuggingFace model with optional quantization

    Args:
        model_name: HuggingFace model name or path
        device: Device to load on ("auto", "cuda", "cpu")
        use_8bit: Use 8-bit quantization
        use_4bit: Use 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        # Configure quantization
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load tokenizer
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        print(f"Loading model {model_name}...")
        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": device,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        print(f"Model loaded successfully on {device}")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    device: str = "cuda"
) -> str:
    """
    Generate text using a loaded model

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        device: Device for inference

    Returns:
        Generated text
    """
    try:
        import torch

        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"Error generating text: {e}")
        raise


def print_system_info():
    """Print system information for debugging"""
    print("=" * 50)
    print("System Information")
    print("=" * 50)

    # Python version
    print(f"Python: {sys.version}")

    # PyTorch info
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")

    # Transformers info
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")

    # LangChain info
    try:
        import langchain
        print(f"LangChain: {langchain.__version__}")
    except ImportError:
        print("LangChain: Not installed")

    # CrewAI info
    try:
        import crewai
        print(f"CrewAI: {crewai.__version__}")
    except ImportError:
        print("CrewAI: Not installed")

    # Ollama info
    print(f"Ollama Available: {check_ollama_available()}")
    if check_ollama_available():
        models = get_ollama_models()
        print(f"Ollama Models: {', '.join(models) if models else 'None'}")

    # Environment variables
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")

    print("=" * 50)


if __name__ == "__main__":
    print_system_info()
