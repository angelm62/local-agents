"""
LangChain Batch Processing with Async Execution
Demonstrates parallel processing of multiple inputs
"""

import sys
import os
import asyncio
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from shared.utils import check_ollama_available, get_ollama_models

# Load environment variables
load_dotenv()


async def process_batch_async():
    """Process multiple inputs asynchronously"""
    print("=" * 70)
    print("LangChain Batch Processing (Async)")
    print("=" * 70)
    print()

    # Check Ollama availability
    if not check_ollama_available():
        print("Error: Ollama server not available")
        print("Please start Ollama with: ollama serve")
        sys.exit(1)

    # Get available models
    models = get_ollama_models()
    if not models:
        print("Error: No Ollama models found")
        sys.exit(1)

    model_name = models[0]
    print(f"Using model: {model_name}")
    print()

    try:
        from langchain_community.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        # Initialize Ollama LLM
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.7
        )

        # Create prompt template
        template = """Summarize the following text in one sentence:

Text: {text}

Summary:"""

        prompt = PromptTemplate(
            input_variables=["text"],
            template=template
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        # Batch inputs
        texts = [
            "Artificial intelligence is transforming industries by automating tasks and providing insights.",
            "Machine learning models learn patterns from data to make predictions.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple layers to process complex data.",
            "AI agents can perform tasks autonomously by making decisions based on their environment."
        ]

        print(f"Processing {len(texts)} texts in parallel...")
        print()

        start_time = time.time()

        # Process batch (LangChain batch method)
        results = await chain.abatch([{"text": text} for text in texts])

        elapsed = time.time() - start_time

        # Display results
        print("Results:")
        print("=" * 70)
        for i, (text, result) in enumerate(zip(texts, results), 1):
            print(f"\n{i}. Input: {text[:60]}...")
            print(f"   Summary: {result['text'].strip()}")

        print()
        print("=" * 70)
        print(f"Processed {len(texts)} texts in {elapsed:.2f} seconds")
        print(f"Average: {elapsed/len(texts):.2f} seconds per text")
        print()
        print("Batch processing completed successfully!")

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def process_batch_sequential():
    """Process multiple inputs sequentially for comparison"""
    print("=" * 70)
    print("LangChain Batch Processing (Sequential for comparison)")
    print("=" * 70)
    print()

    # Check Ollama availability
    if not check_ollama_available():
        print("Error: Ollama server not available")
        sys.exit(1)

    models = get_ollama_models()
    if not models:
        print("Error: No Ollama models found")
        sys.exit(1)

    model_name = models[0]
    print(f"Using model: {model_name}")
    print()

    try:
        from langchain_community.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.7
        )

        template = """Summarize in one sentence: {text}"""
        prompt = PromptTemplate(input_variables=["text"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt)

        texts = [
            "AI is transforming industries.",
            "ML learns from data patterns.",
            "NLP processes human language."
        ]

        print(f"Processing {len(texts)} texts sequentially...")
        start_time = time.time()

        results = [chain.run(text=text) for text in texts]

        elapsed = time.time() - start_time

        print(f"Sequential processing took {elapsed:.2f} seconds")
        print(f"Average: {elapsed/len(texts):.2f} seconds per text")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run async batch processing
    asyncio.run(process_batch_async())

    print("\n" + "=" * 70 + "\n")

    # Run sequential for comparison
    # process_batch_sequential()
