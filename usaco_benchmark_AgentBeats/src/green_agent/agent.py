"""Green agent implementation - manages USACO assessment and evaluation."""

import uvicorn
import sys
import dotenv
import json
import time
import os
import subprocess
import threading
import psutil

# Handle tomllib import for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util.parse_tags import parse_tags
from src.my_util import my_a2a

dotenv.load_dotenv()


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def _find_test_cases(problem_dir: str) -> list:
    """Finds pairs of input/output files (e.g., ride.in, ride.out)."""
    files = os.listdir(problem_dir)
    in_files = sorted([f for f in files if f.endswith('.in')])
    test_cases = []
    for in_file in in_files:
        base_name = in_file.rsplit('.', 1)[0]
        out_file = f"{base_name}.out"
        if out_file in files:
            test_cases.append((os.path.join(problem_dir, in_file), os.path.join(problem_dir, out_file)))
    return test_cases


def _monitor_memory(process: subprocess.Popen, results_dict: dict):
    """
    A target function for a thread that polls the memory usage of a process.
    """
    peak_memory_mb = 0
    results_dict['peak_memory_mb'] = 0
    try:
        p = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                memory_bytes = p.memory_info().rss
                peak_memory_mb = max(peak_memory_mb, memory_bytes / (1024 * 1024))
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass
    finally:
        results_dict['peak_memory_mb'] = peak_memory_mb


def _run_single_test(script_path: str, input_data: str, timeout: int = 2) -> dict:
    """Executes the solution against a single input."""
    result = {}

    start_time = time.perf_counter()

    try:
        process = subprocess.Popen(
            ['python', script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        mem_thread = threading.Thread(target=_monitor_memory, args=(process, result))
        mem_thread.start()

        stdout, stderr = process.communicate(input=input_data, timeout=timeout)

        mem_thread.join()

        end_time = time.perf_counter()

        result['runtime_ms'] = (end_time - start_time) * 1000
        result['output'] = stdout

        if process.returncode != 0:
            result['status'] = 'Runtime Error'
            result['error_details'] = stderr
        else:
            result['status'] = 'Completed'

    except subprocess.TimeoutExpired:
        process.kill()
        mem_thread.join()
        result['status'] = 'Timeout'
        result['runtime_ms'] = timeout * 1000

    except Exception as e:
        result['status'] = 'Execution Error'
        result['error_details'] = str(e)

    return result


async def ask_agent_to_solve_problem(white_agent_url: str, problem_name: str, benchmark_path: str, timeout_seconds: int = 2):
    """
    Ask the white agent to solve a USACO problem by providing the problem description
    and iteratively providing test case feedback.
    """
    problem_dir = os.path.join(benchmark_path, 'problems', problem_name)
    if not os.path.isdir(problem_dir):
        return {"error": f"Problem '{problem_name}' not found."}

    test_cases = _find_test_cases(problem_dir)
    if not test_cases:
        return {"error": f"No test cases found in '{problem_dir}'."}

    # Read problem description if available
    problem_desc_file = os.path.join(problem_dir, f"{problem_name}.txt")
    if os.path.exists(problem_desc_file):
        with open(problem_desc_file, 'r') as f:
            problem_description = f.read()
    else:
        # Generate a simple description from test cases
        with open(test_cases[0][0], 'r') as f:
            sample_input = f.read()
        with open(test_cases[0][1], 'r') as f:
            sample_output = f.read()
        problem_description = f"""USACO Problem: {problem_name}

Sample Input:
{sample_input}

Expected Output:
{sample_output}

Please write a Python solution that reads from stdin and writes to stdout.
Your solution should be wrapped in <solution>...</solution> tags.
"""

    # Send initial problem description
    print(f"@@@ Green agent: Sending problem description to white agent...")
    initial_message = f"""You need to solve the following USACO programming problem:

{problem_description}

Please provide your Python solution wrapped in <solution>...</solution> tags.
The solution should read input from stdin and write output to stdout.
"""

    white_agent_response = await my_a2a.send_message(
        white_agent_url, initial_message, context_id=None
    )
    res_root = white_agent_response.root
    assert isinstance(res_root, SendMessageSuccessResponse)
    res_result = res_root.result
    assert isinstance(res_result, Message)

    context_id = res_result.context_id
    text_parts = get_text_parts(res_result.parts)
    assert len(text_parts) == 1
    white_text = text_parts[0]
    print(f"@@@ White agent response:\n{white_text}")

    # Parse the solution
    white_tags = parse_tags(white_text)
    if "solution" not in white_tags:
        return {
            "error": "White agent did not provide solution in required format",
            "response": white_text
        }

    solution_code = white_tags["solution"]

    # Save the solution to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        solution_path = f.name
        f.write(solution_code)

    try:
        # Run the solution against all test cases
        print(f"\nEvaluating '{problem_name}' with white agent's solution...")

        case_results = []
        overall_passed = 0

        for i, (in_file, out_file) in enumerate(test_cases, 1):
            with open(in_file, 'r') as f_in:
                input_data = f_in.read()
            with open(out_file, 'r') as f_out:
                expected_output = f_out.read().strip()

            result = _run_single_test(solution_path, input_data, timeout_seconds)

            if result['status'] == 'Completed':
                actual_output = result['output'].strip()
                if actual_output == expected_output:
                    result['verdict'] = 'Correct'
                    overall_passed += 1
                else:
                    result['verdict'] = 'Wrong Answer'
            else:
                result['verdict'] = result['status']

            print(f"  - Test Case #{i}: {result['verdict']} ({result['runtime_ms']:.2f} ms, {result['peak_memory_mb']:.2f} MB)")
            case_results.append(result)

        final_report = {
            "problem": problem_name,
            "summary": {
                "total_cases": len(test_cases),
                "passed_cases": overall_passed,
                "accuracy": f"{overall_passed / len(test_cases) * 100:.2f}%",
                "avg_runtime_ms": sum(r['runtime_ms'] for r in case_results) / len(case_results),
                "max_memory_mb": max(r['peak_memory_mb'] for r in case_results)
            },
            "details": case_results
        }

        return final_report

    finally:
        # Clean up temporary file
        os.unlink(solution_path)


class USACOGreenAgentExecutor(AgentExecutor):
    def __init__(self, benchmark_path: str = "."):
        self.benchmark_path = benchmark_path

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Parse the task
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        problem_config_str = tags["problem_config"]
        problem_config = json.loads(problem_config_str)

        print("Green agent: Starting USACO evaluation...")
        timestamp_started = time.time()

        problem_name = problem_config["problem_name"]
        timeout_seconds = problem_config.get("timeout_seconds", 2)

        res = await ask_agent_to_solve_problem(
            white_agent_url,
            problem_name,
            self.benchmark_path,
            timeout_seconds
        )

        metrics = {}
        metrics["time_used"] = time.time() - timestamp_started

        if "error" in res:
            result_emoji = "❌"
            metrics["success"] = False
            metrics["error"] = res["error"]
        else:
            result_bool = metrics["success"] = (res["summary"]["passed_cases"] == res["summary"]["total_cases"])
            result_emoji = "✅" if result_bool else "❌"
            metrics["accuracy"] = res["summary"]["accuracy"]
            metrics["passed_cases"] = res["summary"]["passed_cases"]
            metrics["total_cases"] = res["summary"]["total_cases"]

        print("Green agent: Evaluation complete.")
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished. White agent success: {result_emoji}\nMetrics: {json.dumps(metrics, indent=2)}\nFull Report: {json.dumps(res, indent=2)}\n"
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="usaco_green_agent", host="localhost", port=9001, benchmark_path="."):
    print("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    request_handler = DefaultRequestHandler(
        agent_executor=USACOGreenAgentExecutor(benchmark_path=benchmark_path),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
