"""Green agent implementation - manages USACO problem assessment and evaluation."""

import uvicorn
import tomllib
import dotenv
import json
import os
import subprocess
import time
import threading
import psutil
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util import parse_tags, my_a2a

dotenv.load_dotenv()


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def _monitor_memory(process: subprocess.Popen, results_dict: dict):
    """Monitor memory usage of a process."""
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
    """Execute the solution against a single input."""
    result = {}
    start_time = time.perf_counter()
    
    try:
        process = subprocess.Popen(
            ['python3', script_path],
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
        result['output'] = ''
    
    except Exception as e:
        end_time = time.perf_counter()
        result['status'] = 'Execution Error'
        result['error_details'] = str(e)
        result['runtime_ms'] = (end_time - start_time) * 1000
        result['output'] = f'Execution Error: {str(e)}'

    if 'peak_memory_mb' not in result:
        result['peak_memory_mb'] = 0

    return result


def _find_test_cases(problem_dir: str) -> list:
    """Find pairs of input/output files."""
    files = os.listdir(problem_dir)
    in_files = sorted([f for f in files if f.endswith('.in')])
    test_cases = []
    for in_file in in_files:
        base_name = in_file.rsplit('.', 1)[0]
        out_file = f"{base_name}.out"
        if out_file in files:
            test_cases.append((
                os.path.join(problem_dir, in_file),
                os.path.join(problem_dir, out_file)
            ))
    return test_cases


async def evaluate_solution_with_white_agent(
    white_agent_url: str,
    problem_name: str,
    benchmark_path: str,
    timeout: int = 2
) -> dict:
    """Request white agent to generate code and evaluate it."""
    
    problem_dir = os.path.join(benchmark_path, 'problems', problem_name)
    if not os.path.isdir(problem_dir):
        return {"error": f"Problem '{problem_name}' not found."}
    
    # Read problem description
    problem_desc_path = os.path.join(problem_dir, 'description.txt')
    if not os.path.isfile(problem_desc_path):
        return {"error": f"Problem description not found for '{problem_name}'."}
    
    with open(problem_desc_path, 'r') as f:
        description = f.read()
    
    # Request white agent to generate solution
    task_description = f"""
Please generate a Python solution for the following USACO problem.
The solution should read from stdin and write to stdout.
Provide ONLY the Python code, no explanations or markdown formatting.

<problem_description>
{description}
</problem_description>

Please wrap your Python code with <code>...</code> tags.
"""
    
    print(f"@@@ Green agent: Requesting solution from white agent for problem '{problem_name}'...")
    
    white_agent_response = await my_a2a.send_message(white_agent_url, task_description)
    res_root = white_agent_response.root
    from a2a.types import SendMessageSuccessResponse, Message
    assert isinstance(res_root, SendMessageSuccessResponse)
    res_result = res_root.result
    assert isinstance(res_result, Message)
    
    text_parts = get_text_parts(res_result.parts)
    assert len(text_parts) == 1, "Expecting exactly one text part from the white agent"
    white_text = text_parts[0]
    
    # Parse code from response
    white_tags = parse_tags(white_text)
    if 'code' not in white_tags:
        return {"error": "White agent did not provide code in expected format."}
    
    generated_code = white_tags['code']
    
    # Save the generated code
    solution_dir = os.path.join(benchmark_path, 'solutions_agent')
    os.makedirs(solution_dir, exist_ok=True)
    solution_path = os.path.join(solution_dir, f"{problem_name}_generated.py")
    
    with open(solution_path, 'w') as f:
        f.write(generated_code)
    
    print(f"@@@ Green agent: Code saved to {solution_path}")
    
    # Find test cases
    test_cases = _find_test_cases(problem_dir)
    if not test_cases:
        return {"error": f"No test cases found in '{problem_dir}'."}
    
    # Run evaluation
    print(f"@@@ Green agent: Evaluating solution against {len(test_cases)} test cases...")
    
    case_results = []
    overall_passed = 0

    for i, (in_file, out_file) in enumerate(test_cases, 1):
        with open(in_file, 'r') as f_in:
            input_data = f_in.read()
        with open(out_file, 'r') as f_out:
            expected_output = f_out.read().strip()

        result = _run_single_test(solution_path, input_data, timeout)
        result['expected'] = expected_output
        result['input_preview'] = input_data[:300]
        
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

    # Calculate metrics
    total_cases = len(test_cases)
    accuracy_pct = (overall_passed / total_cases) * 100 if total_cases else 0.0
    avg_runtime_ms = sum(r['runtime_ms'] for r in case_results) / len(case_results)
    max_memory_mb = max(r['peak_memory_mb'] for r in case_results)

    timeout_ms = timeout * 1000
    runtime_percent = max(0.0, min(100.0, (timeout_ms - avg_runtime_ms) / timeout_ms * 100.0))
    memory_threshold_mb = 256.0
    memory_percent = max(0.0, min(100.0, (memory_threshold_mb - max_memory_mb) / memory_threshold_mb * 100.0))

    final_percent = (accuracy_pct * 0.7) + (runtime_percent * 0.2) + (memory_percent * 0.1)

    # Generate suggestions
    suggestions = []
    if accuracy_pct < 100.0:
        suggestions.append("Some test cases failed. Check logic and edge cases.")
    if any(r.get('status') == 'Timeout' for r in case_results):
        suggestions.append("Timeout detected - consider optimizing algorithm.")
    if any(r.get('status') == 'Runtime Error' for r in case_results):
        suggestions.append("Runtime errors occurred. Check exception messages.")
    if not suggestions:
        suggestions.append("All tests passed!")

    final_report = {
        "problem": problem_name,
        "solution": os.path.basename(solution_path),
        "summary": {
            "total_cases": total_cases,
            "passed_cases": overall_passed,
            "accuracy": f"{accuracy_pct:.2f}%",
            "avg_runtime_ms": round(avg_runtime_ms, 2),
            "max_memory_mb": round(max_memory_mb, 2)
        },
        "analysis": {
            'accuracy_pct': round(accuracy_pct, 2),
            'runtime_ms_avg': round(avg_runtime_ms, 2),
            'max_memory_mb': round(max_memory_mb, 2),
            'runtime_percent': round(runtime_percent, 2),
            'memory_percent': round(memory_percent, 2),
            'final_score_percent': round(final_percent, 2),
            'suggestions': suggestions,
        },
        "details": case_results
    }

    return final_report


class UsacoGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        
        white_agent_url = tags.get("white_agent_url")
        problem_config_str = tags.get("problem_config")
        
        if not white_agent_url or not problem_config_str:
            await event_queue.enqueue_event(
                new_agent_text_message("Error: Missing required configuration.")
            )
            return
        
        problem_config = json.loads(problem_config_str)
        problem_name = problem_config.get("problem_name")
        benchmark_path = problem_config.get("benchmark_path", ".")
        timeout = problem_config.get("timeout", 2)
        
        print(f"Green agent: Evaluating problem '{problem_name}'...")
        
        timestamp_started = time.time()
        result = await evaluate_solution_with_white_agent(
            white_agent_url, problem_name, benchmark_path, timeout
        )
        
        time_used = time.time() - timestamp_started
        
        if "error" in result:
            message = f"❌ Evaluation failed: {result['error']}"
        else:
            success = result['analysis']['accuracy_pct'] == 100.0
            result_emoji = "✅" if success else "❌"
            message = f"""Evaluation complete {result_emoji}

Problem: {result['problem']}
Time used: {time_used:.2f}s

Summary:
- Passed: {result['summary']['passed_cases']}/{result['summary']['total_cases']}
- Accuracy: {result['summary']['accuracy']}
- Avg Runtime: {result['summary']['avg_runtime_ms']} ms
- Max Memory: {result['summary']['max_memory_mb']} MB
- Final Score: {result['analysis']['final_score_percent']}%

Suggestions:
{chr(10).join('- ' + s for s in result['analysis']['suggestions'])}
"""
        
        print("Green agent: Evaluation complete.")
        await event_queue.enqueue_event(new_agent_text_message(message))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="usaco_green_agent", host="localhost", port=8010):
    print("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    request_handler = DefaultRequestHandler(
        agent_executor=UsacoGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
