"""
CrewAI Tools Demonstration
Shows how agents can use custom Python functions as tools
"""

import sys
import os
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from shared.utils import check_ollama_available, get_ollama_models

# Load environment variables
load_dotenv()


# Custom tool functions
def calculate_sum(numbers: str) -> str:
    """Calculate the sum of comma-separated numbers"""
    try:
        nums = [float(n.strip()) for n in numbers.split(',')]
        result = sum(nums)
        return f"The sum of {numbers} is {result}"
    except Exception as e:
        return f"Error calculating sum: {e}"


def get_current_time() -> str:
    """Get the current date and time"""
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def word_count(text: str) -> str:
    """Count words in the provided text"""
    words = len(text.split())
    chars = len(text)
    return f"Text statistics: {words} words, {chars} characters"


def main():
    print("=" * 70)
    print("CrewAI Tools Demonstration")
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
        from crewai import Agent, Task, Crew, Process
        from crewai_tools import tool
        from langchain_community.llms import Ollama

        # Initialize Ollama LLM
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = Ollama(
            model=model_name,
            base_url=ollama_base_url
        )

        print("Creating custom tools...")

        # Wrap functions as CrewAI tools
        @tool("Calculator")
        def calculator_tool(numbers: str) -> str:
            """Calculate the sum of comma-separated numbers. Input format: '1,2,3,4'"""
            return calculate_sum(numbers)

        @tool("TimeChecker")
        def time_tool() -> str:
            """Get the current date and time"""
            return get_current_time()

        @tool("WordCounter")
        def word_counter_tool(text: str) -> str:
            """Count words and characters in the provided text"""
            return word_count(text)

        print("Custom tools created:")
        print("  1. Calculator - Sum numbers")
        print("  2. TimeChecker - Get current time")
        print("  3. WordCounter - Count words and characters")
        print()

        # Create agent with tools
        assistant = Agent(
            role='General Assistant',
            goal='Help with various tasks using available tools',
            backstory="""You are a helpful assistant with access to various tools.
            You use the appropriate tools to complete tasks accurately. Always use
            tools when they are relevant to the task.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[calculator_tool, time_tool, word_counter_tool]
        )

        print("Agent created with tools access")
        print()

        # Define tasks that require tool usage
        task1 = Task(
            description="""Use the Calculator tool to find the sum of these numbers:
            15, 27, 33, 41, 8, 19

            Report the result clearly.""",
            agent=assistant,
            expected_output="The sum of the given numbers"
        )

        task2 = Task(
            description="""Use the TimeChecker tool to get the current date and time,
            then report it in a friendly message.""",
            agent=assistant,
            expected_output="Current date and time in a friendly format"
        )

        task3 = Task(
            description="""Use the WordCounter tool to analyze this text:
            'Artificial intelligence and machine learning are transforming how we build software systems.'

            Report the word count and character count.""",
            agent=assistant,
            expected_output="Word and character count of the text"
        )

        print("Tasks created (each requiring tool usage)")
        print()

        # Create crew
        crew = Crew(
            agents=[assistant],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=2
        )

        print("Crew assembled")
        print()
        print("=" * 70)
        print("Starting crew with tools execution...")
        print("=" * 70)
        print()

        # Execute crew
        result = crew.kickoff()

        print()
        print("=" * 70)
        print("Tools Demonstration Complete!")
        print("=" * 70)
        print()
        print("Final Results:")
        print("-" * 70)
        print(result)
        print()
        print("-" * 70)
        print("The agent successfully used all three custom tools!")

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please ensure crewai-tools is installed: pip install crewai-tools")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
