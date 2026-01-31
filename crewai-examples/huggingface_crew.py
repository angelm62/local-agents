"""
CrewAI with HuggingFace Local Models
Demonstrates CrewAI powered by local HuggingFace models with GPU acceleration
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
    print("CrewAI with Local HuggingFace Models")
    print("=" * 70)
    print()

    # Check GPU
    device = get_optimal_device()
    print(f"Using device: {device}")
    print()

    try:
        from crewai import Agent, Task, Crew, Process
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        # Use small model for testing
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Loading model: {model_name}")
        print("This may take a few minutes on first run...")
        print()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        print("Loading model to device...")
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

        # Create pipeline
        print("Creating text generation pipeline...")
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
        print("Pipeline created successfully")
        print()

        # Create agents
        print("Creating crew with HuggingFace-powered agents...")

        researcher = Agent(
            role='AI Researcher',
            goal='Research and explain AI concepts clearly',
            backstory="""You are an AI researcher who specializes in explaining
            complex concepts in simple terms.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        analyst = Agent(
            role='Technology Analyst',
            goal='Analyze technology trends and provide insights',
            backstory="""You are a technology analyst who identifies key insights
            and trends in the tech industry.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        print("Agents created successfully")
        print()

        # Define tasks
        task1 = Task(
            description="""Explain what large language models (LLMs) are in 2-3 sentences.
            Focus on the core concept and main capabilities.""",
            agent=researcher,
            expected_output="A clear 2-3 sentence explanation of LLMs"
        )

        task2 = Task(
            description="""Based on the explanation of LLMs, identify 2 key applications
            where they are being used today. Be specific and concise.""",
            agent=analyst,
            expected_output="2 specific applications of LLMs"
        )

        print("Tasks defined")
        print()

        # Create crew
        crew = Crew(
            agents=[researcher, analyst],
            tasks=[task1, task2],
            process=Process.sequential,
            verbose=2
        )

        print("Crew assembled")
        print()
        print("=" * 70)
        print("Starting HuggingFace-powered crew execution...")
        print("Note: Local models may take longer to respond than cloud APIs")
        print("=" * 70)
        print()

        # Execute crew
        result = crew.kickoff()

        print()
        print("=" * 70)
        print("Execution Complete!")
        print("=" * 70)
        print()
        print("Final Result:")
        print("-" * 70)
        print(result)
        print()
        print("-" * 70)
        print(f"Successfully ran CrewAI with local HuggingFace model on {device}")

        if device.startswith("cuda"):
            print()
            print("GPU was utilized for inference!")
            from shared.utils import get_gpu_info
            gpu_info = get_gpu_info()
            if gpu_info.get("available"):
                print(f"GPU Memory Used: {gpu_info['memory_allocated']:.2f} GB")

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
