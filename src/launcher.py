"""Launcher module - initiates and coordinates the USACO evaluation process."""

import multiprocessing
import json
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.my_util import my_a2a


async def launch_evaluation(
    problems=None,
    benchmark_path=".",
    timeout=2
):
    """Launch the complete USACO evaluation workflow."""
    
    if problems is None:
        problems = ["ride", "gift1", "friday"]
    
    # Start green agent
    print("Launching green agent...")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("usaco_green_agent", *green_address)
    )
    p_green.start()
    assert await my_a2a.wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # Start white agent
    print("Launching white agent...")
    white_address = ("localhost", 9002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    p_white = multiprocessing.Process(
        target=start_white_agent, args=("usaco_white_agent", *white_address)
    )
    p_white.start()
    assert await my_a2a.wait_agent_ready(white_url), "White agent not ready in time"
    print("White agent is ready.")

    # Evaluate each problem
    print(f"\nStarting evaluation of {len(problems)} problems...")
    print("=" * 60)
    
    for i, problem in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}] Evaluating problem: {problem}")
        print("-" * 60)
        
        problem_config = {
            "problem_name": problem,
            "benchmark_path": benchmark_path,
            "timeout": timeout,
        }
        
        task_text = f"""
Your task is to evaluate the USACO problem using the code generation agent located at:
<white_agent_url>
http://{white_address[0]}:{white_address[1]}/
</white_agent_url>

You should use the following problem configuration:
<problem_config>
{json.dumps(problem_config, indent=2)}
</problem_config>
        """
        
        print("Sending task to green agent...")
        response = await my_a2a.send_message(green_url, task_text)
        print("\nResponse from green agent:")
        print(response)
        print("=" * 60)

    print("\n\nEvaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    print("Agents terminated.")


async def launch_remote_evaluation(
    green_url: str,
    white_url: str,
    problems=None,
    benchmark_path=".",
    timeout=2
):
    """Launch evaluation with remote agents (already running)."""
    
    if problems is None:
        problems = ["ride", "gift1", "friday"]
    
    # Check if agents are ready
    print(f"Checking green agent at {green_url}...")
    assert await my_a2a.wait_agent_ready(green_url), "Green agent not ready"
    print("Green agent is ready.")
    
    print(f"Checking white agent at {white_url}...")
    assert await my_a2a.wait_agent_ready(white_url), "White agent not ready"
    print("White agent is ready.")

    # Evaluate each problem
    print(f"\nStarting evaluation of {len(problems)} problems...")
    print("=" * 60)
    
    for i, problem in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}] Evaluating problem: {problem}")
        print("-" * 60)
        
        problem_config = {
            "problem_name": problem,
            "benchmark_path": benchmark_path,
            "timeout": timeout,
        }
        
        task_text = f"""
Your task is to evaluate the USACO problem using the code generation agent located at:
<white_agent_url>
{white_url}
</white_agent_url>

You should use the following problem configuration:
<problem_config>
{json.dumps(problem_config, indent=2)}
</problem_config>
        """
        
        print("Sending task to green agent...")
        response = await my_a2a.send_message(green_url, task_text)
        print("\nResponse from green agent:")
        print(response)
        print("=" * 60)

    print("\n\nEvaluation complete.")
    print("Note: Remote agents are still running. Stop them manually if needed.")
