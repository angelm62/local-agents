"""
CrewAI Basic Crew with Ollama
Simple crew with a single agent using local Ollama models
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
    print("CrewAI Basic Crew with Ollama")
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
        from crewai import Agent, Task, Crew, Process
        from langchain_community.llms import Ollama

        # Initialize Ollama LLM
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = Ollama(
            model=model_name,
            base_url=ollama_base_url
        )

        print("Creating research agent...")

        # Define agent
        researcher = Agent(
            role='Research Analyst',
            goal='Provide detailed and accurate information on any topic',
            backstory="""You are an experienced research analyst with expertise in
            gathering and synthesizing information. You provide clear, concise, and
            well-structured responses based on your knowledge.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        print("Agent created successfully")
        print()

        # Define tasks
        task1 = Task(
            description="""Research and explain the key benefits of using AI agents
            in business automation. Provide at least 3 specific benefits with brief
            explanations.""",
            agent=researcher,
            expected_output="A list of 3+ key benefits with explanations"
        )

        task2 = Task(
            description="""Summarize the main differences between LangChain and CrewAI
            frameworks for building AI agents.""",
            agent=researcher,
            expected_output="A clear comparison of LangChain vs CrewAI"
        )

        print("Tasks created successfully")
        print()

        # Create crew
        crew = Crew(
            agents=[researcher],
            tasks=[task1, task2],
            process=Process.sequential,
            verbose=2
        )

        print("Crew assembled successfully")
        print()
        print("=" * 70)
        print("Starting crew execution...")
        print("=" * 70)
        print()

        # Execute crew
        result = crew.kickoff()

        print()
        print("=" * 70)
        print("Crew Execution Complete!")
        print("=" * 70)
        print()
        print("Final Result:")
        print("-" * 70)
        print(result)
        print()

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please ensure CrewAI is installed: pip install crewai")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
