"""
LangChain Basic Agent with Ollama
Simple conversational agent using local Ollama models
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from shared.utils import check_ollama_available, get_ollama_models

# Load environment variables
load_dotenv()


def main():
    print("=" * 70)
    print("LangChain Basic Agent with Ollama")
    print("=" * 70)
    print()

    # Check Ollama availability
    if not check_ollama_available():
        print("Error: Ollama server not available")
        print("Please start Ollama with: ollama serve")
        print("And ensure you have models installed: ollama list")
        sys.exit(1)

    # Get available models
    models = get_ollama_models()
    if not models:
        print("Error: No Ollama models found")
        print("Install a model with: ollama pull llama2")
        sys.exit(1)

    print(f"Available Ollama models: {', '.join(models)}")
    print()

    # Use first available model
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

        print("LLM initialized successfully")
        print()

        # Create a simple prompt template
        template = """You are a helpful AI assistant. Answer the following question concisely.

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            input_variables=["question"],
            template=template
        )

        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Test questions
        questions = [
            "What is the capital of France?",
            "Explain what machine learning is in one sentence.",
            "What are the benefits of using AI agents?"
        ]

        print("Running test questions...")
        print("=" * 70)
        print()

        for i, question in enumerate(questions, 1):
            print(f"Question {i}: {question}")
            print("-" * 70)

            response = chain.run(question=question)
            print(f"Answer: {response.strip()}")
            print()

        print("=" * 70)
        print("Basic agent test completed successfully!")

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please ensure LangChain is installed: pip install langchain langchain-community")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
