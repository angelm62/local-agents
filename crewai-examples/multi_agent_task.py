"""
CrewAI Multi-Agent Collaboration
Demonstrates multiple agents working together on a complex task
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
    print("CrewAI Multi-Agent Collaboration")
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
        from langchain_community.llms import Ollama

        # Initialize Ollama LLM
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = Ollama(
            model=model_name,
            base_url=ollama_base_url
        )

        print("Creating multi-agent crew...")
        print()

        # Agent 1: Researcher
        researcher = Agent(
            role='Technology Researcher',
            goal='Research and gather information about AI technologies',
            backstory="""You are a technology researcher with deep knowledge of AI
            and machine learning. You excel at finding and organizing information
            about technical topics.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # Agent 2: Analyst
        analyst = Agent(
            role='Technical Analyst',
            goal='Analyze technical information and identify key insights',
            backstory="""You are a technical analyst who excels at breaking down
            complex topics into understandable insights. You identify trends,
            benefits, and potential challenges.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # Agent 3: Writer
        writer = Agent(
            role='Technical Writer',
            goal='Create clear and engaging technical documentation',
            backstory="""You are a skilled technical writer who transforms complex
            technical information into clear, concise, and engaging content suitable
            for various audiences.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        print("Agents created:")
        print(f"  1. {researcher.role}")
        print(f"  2. {analyst.role}")
        print(f"  3. {writer.role}")
        print()

        # Define sequential tasks
        research_task = Task(
            description="""Research the topic of 'AI Agents in Software Development'.
            Gather information about:
            1. What AI agents are
            2. How they are used in software development
            3. Popular frameworks (mention LangChain and CrewAI)

            Provide a structured summary of your findings.""",
            agent=researcher,
            expected_output="Structured research summary covering all 3 points"
        )

        analysis_task = Task(
            description="""Based on the research, analyze:
            1. The main benefits of using AI agents in development
            2. Potential challenges or limitations
            3. Future trends in this space

            Provide clear insights for each area.""",
            agent=analyst,
            expected_output="Analysis with benefits, challenges, and trends"
        )

        writing_task = Task(
            description="""Using the research and analysis, write a brief article
            (300-400 words) titled 'AI Agents: Transforming Software Development'.

            The article should:
            - Start with an engaging introduction
            - Explain what AI agents are
            - Highlight key benefits
            - Mention popular frameworks
            - End with a forward-looking conclusion

            Write in a clear, professional yet accessible style.""",
            agent=writer,
            expected_output="A 300-400 word article with proper structure"
        )

        print("Tasks defined:")
        print(f"  1. Research Task → {researcher.role}")
        print(f"  2. Analysis Task → {analyst.role}")
        print(f"  3. Writing Task → {writer.role}")
        print()

        # Create crew with sequential process
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.sequential,
            verbose=2
        )

        print("Crew assembled with sequential workflow")
        print()
        print("=" * 70)
        print("Starting multi-agent collaboration...")
        print("This will take a few minutes as each agent completes their task...")
        print("=" * 70)
        print()

        # Execute crew
        result = crew.kickoff()

        print()
        print("=" * 70)
        print("Multi-Agent Collaboration Complete!")
        print("=" * 70)
        print()
        print("Final Article:")
        print("-" * 70)
        print(result)
        print()
        print("-" * 70)
        print("This article was created through collaboration of 3 specialized agents!")

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
