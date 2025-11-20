"""Launcher module - initiates and coordinates the evaluation process."""

import multiprocessing
import json
import os
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.my_util import my_a2a


async def launch_evaluation(problem_name="ride", timeout_seconds=2):
    # Get the benchmark root path (parent of src directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.dirname(current_dir)

    # Start green agent
    print("Launching green agent...")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("usaco_green_agent", green_address[0], green_address[1], benchmark_path)
    )
    p_green.start()
    assert await my_a2a.wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # Start white agent
    print("Launching white agent...")
    white_address = ("localhost", 9002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    p_white = multiprocessing.Process(
        target=start_white_agent, args=("general_white_agent", white_address[0], white_address[1])
    )
    p_white.start()
    assert await my_a2a.wait_agent_ready(white_url), "White agent not ready in time"
    print("White agent is ready.")

    # Send the task description
    print("Sending task description to green agent...")
    task_config = {
        "problem_name": problem_name,
        "timeout_seconds": timeout_seconds,
    }
    task_text = f"""
Your task is to evaluate a USACO problem solution from the agent located at:
<white_agent_url>
http://{white_address[0]}:{white_address[1]}/
</white_agent_url>
You should use the following problem configuration:
<problem_config>
{json.dumps(task_config, indent=2)}
</problem_config>
    """
    print("Task description:")
    print(task_text)
    print("Sending...")
    response = await my_a2a.send_message(green_url, task_text)
    print("Response from green agent:")
    print(response)

    print("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    print("Agents terminated.")
