"""
LangChain with Local HuggingFace Models
Demonstrates using local HuggingFace models with GPU acceleration
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from shared.utils import check_gpu_available, get_optimal_device

# Load environment variables
load_dotenv()


def main():
    print("=" * 70)
    print("LangChain with Local HuggingFace Models")
    print("=" * 70)
    print()

    # Check GPU
    device = get_optimal_device()
    print(f"Using device: {device}")
    print()

    try:
        from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        # Choose model based on GPU availability
        if device.startswith("cuda"):
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print("GPU available - using TinyLlama 1.1B")
        else:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print("CPU mode - using TinyLlama 1.1B")

        print(f"Loading model: {model_name}")
        print()

        # Load model and tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map=device if device.startswith("cuda") else None,
            low_cpu_mem_usage=True
        )

        if device == "cpu":
            model = model.to(device)

        print("Model loaded successfully")
        print()

        # Create text generation pipeline
        print("Creating pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            device=0 if device.startswith("cuda") else -1
        )

        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)

        # Create prompt template
        template = """<|system|>
You are a helpful AI assistant. Provide clear and concise answers.</s>
<|user|>
{question}</s>
<|assistant|>
"""

        prompt = PromptTemplate(
            input_variables=["question"],
            template=template
        )

        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)

        print("Pipeline created successfully")
        print()

        # Test questions
        questions = [
            "What are the main benefits of using AI in healthcare?",
            "Explain the difference between supervised and unsupervised learning.",
            "What is a neural network?"
        ]

        print("Running test questions...")
        print("=" * 70)
        print()

        for i, question in enumerate(questions, 1):
            print(f"Question {i}: {question}")
            print("-" * 70)

            response = chain.run(question=question)

            # Extract just the answer part
            if "<|assistant|>" in response:
                answer = response.split("<|assistant|>")[-1].strip()
            else:
                answer = response.strip()

            # Truncate if too long
            if len(answer) > 300:
                answer = answer[:300] + "..."

            print(f"Answer: {answer}")
            print()

        print("=" * 70)
        print("Test completed successfully!")
        print()

        # Demonstrate embeddings
        print("Bonus: Testing HuggingFace Embeddings...")
        print("-" * 70)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "The weather is nice today."
        ]

        print("Computing embeddings for sample texts...")
        embedded_texts = embeddings.embed_documents(texts)

        print(f"Generated {len(embedded_texts)} embeddings")
        print(f"Embedding dimension: {len(embedded_texts[0])}")

        # Compute similarity (simple dot product)
        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        print()
        print("Similarity scores:")
        print(f"  Text 1 vs Text 2: {cosine_similarity(embedded_texts[0], embedded_texts[1]):.4f}")
        print(f"  Text 1 vs Text 3: {cosine_similarity(embedded_texts[0], embedded_texts[2]):.4f}")
        print(f"  Text 2 vs Text 3: {cosine_similarity(embedded_texts[1], embedded_texts[2]):.4f}")

        print()
        print("Note: Higher scores indicate more semantic similarity")
        print()
        print("=" * 70)
        print("All tests completed successfully!")

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please ensure all packages are installed")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
