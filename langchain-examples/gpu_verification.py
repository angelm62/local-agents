"""
LangChain GPU Verification
Load HuggingFace model on GPU and verify usage with monitoring
"""

import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from shared.utils import check_gpu_available, get_gpu_info
from shared.gpu_monitor import print_gpu_info, get_optimal_device

# Load environment variables
load_dotenv()


def main():
    print("=" * 70)
    print("LangChain GPU Verification with HuggingFace Models")
    print("=" * 70)
    print()

    # Check GPU availability
    print("Step 1: Checking GPU availability...")
    if not check_gpu_available():
        print("Warning: GPU not available. This test requires CUDA.")
        print("Continuing with CPU for demonstration...")
        device = "cpu"
    else:
        device = get_optimal_device()
        print(f"GPU available! Using device: {device}")

    print()
    print("Initial GPU State:")
    print_gpu_info(detailed=True)
    print()

    try:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        # Use a small model for testing
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Step 2: Loading model: {model_name}")
        print("This may take a few minutes on first run (downloading model)...")
        print()

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with GPU support
        print(f"Loading model on {device}...")
        load_start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map=device if device.startswith("cuda") else None,
            low_cpu_mem_usage=True
        )

        if device == "cpu":
            model = model.to(device)

        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f} seconds")
        print()

        # Check GPU state after loading
        if device.startswith("cuda"):
            print("GPU State After Model Loading:")
            print_gpu_info(detailed=True)
            print()

            # Print memory usage
            gpu_info = get_gpu_info()
            if gpu_info.get("available"):
                print(f"GPU Memory Allocated: {gpu_info['memory_allocated']:.2f} GB")
                print(f"GPU Memory Reserved: {gpu_info['memory_reserved']:.2f} GB")
                print()

        # Create pipeline
        print("Step 3: Creating LangChain pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            device=0 if device.startswith("cuda") else -1
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print("Pipeline created successfully")
        print()

        # Test inference
        print("Step 4: Running test inference...")
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning briefly."
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            print("-" * 70)

            inference_start = time.time()
            response = llm.invoke(prompt)
            inference_time = time.time() - inference_start

            print(f"Response: {response[:200]}...")
            print(f"Inference time: {inference_time:.2f} seconds")

            if device.startswith("cuda"):
                gpu_info = get_gpu_info()
                if gpu_info.get("available"):
                    print(f"GPU Memory Used: {gpu_info['memory_allocated']:.2f} GB")

        print()
        print("=" * 70)
        print("GPU Verification Complete!")
        print()

        if device.startswith("cuda"):
            print("Final GPU State:")
            print_gpu_info(detailed=True)
        else:
            print("Test completed on CPU (GPU not available)")

        print()
        print("Summary:")
        print(f"  - Device: {device}")
        print(f"  - Model: {model_name}")
        print(f"  - Load Time: {load_time:.2f}s")
        print(f"  - Test completed successfully")

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please ensure all packages are installed:")
        print("  pip install langchain langchain-huggingface transformers torch")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
