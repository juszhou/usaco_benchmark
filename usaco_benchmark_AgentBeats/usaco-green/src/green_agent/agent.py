"""Green agent implementation - Fully automatic USACO benchmark evaluation.

Green agent automatically evaluates all problems (ride, gift1, friday) when triggered,
without requiring any user input.
"""

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
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util import my_a2a

dotenv.load_dotenv()

# All problems to evaluate automatically
PROBLEMS_TO_TEST = ["ride", "gift1", "friday"]


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
    """Execute solution against a single input."""
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


async def evaluate_single_problem(problem_name: str, white_agent_url: str, benchmark_path: str = ".") -> dict:
    """Evaluate a single problem using white agent."""
    
    problem_dir = os.path.join(benchmark_path, 'problems', problem_name)
    if not os.path.isdir(problem_dir):
        return {"error": f"Problem directory '{problem_dir}' not found"}
    
    # Read problem description
    desc_path = os.path.join(problem_dir, 'description.txt')
    if not os.path.isfile(desc_path):
        return {"error": f"Description file not found: {desc_path}"}
    
    with open(desc_path, 'r') as f:
        description = f.read()
    
    print(f"Green: Requesting code for '{problem_name}' from white agent...")
    
    # Request code from white agent
    try:
        response = await my_a2a.send_message(white_agent_url, description)
        res_root = response.root
        
        if not isinstance(res_root, SendMessageSuccessResponse):
            return {"error": "Invalid response from white agent"}
        
        res_result = res_root.result
        if not isinstance(res_result, Message):
            return {"error": "Invalid message from white agent"}
        
        text_parts = get_text_parts(res_result.parts)
        if not text_parts:
            return {"error": "No text response from white agent"}
        
        code_response = text_parts[0]
        
        # Extract code from markdown if present
        code = code_response.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        # Save generated code
        solution_dir = os.path.join(benchmark_path, 'solutions_agent')
        os.makedirs(solution_dir, exist_ok=True)
        solution_path = os.path.join(solution_dir, f"{problem_name}_generated.py")
        
        with open(solution_path, 'w') as f:
            f.write(code)
        
        print(f"Green: Code saved to {solution_path}")
        
    except Exception as e:
        import traceback
        return {"error": f"Failed to get code from white agent: {str(e)}\n{traceback.format_exc()}"}
    
    # Run tests
    test_cases = _find_test_cases(problem_dir)
    if not test_cases:
        return {"error": f"No test cases found in '{problem_dir}'"}
    
    print(f"Green: Running {len(test_cases)} test cases...")
    
    case_results = []
    overall_passed = 0
    
    for i, (in_file, out_file) in enumerate(test_cases, 1):
        with open(in_file, 'r') as f_in:
            input_data = f_in.read()
        with open(out_file, 'r') as f_out:
            expected_output = f_out.read().strip()

        result = _run_single_test(solution_path, input_data)
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
            result['verdict'] = result['status']  # Timeout Error

        print(f"  - Test Case #{i}: {result['verdict']} ({result['runtime_ms']:.2f} ms, {result['peak_memory_mb']:.2f} MB)")
        case_results.append(result)
    
    # Calculate scores (matching original greenagent.py logic)
    total_cases = len(test_cases)
    accuracy_pct = (overall_passed / total_cases) * 100 if total_cases else 0.0
    avg_runtime_ms = sum(r['runtime_ms'] for r in case_results) / len(case_results)
    max_memory_mb = max(r['peak_memory_mb'] for r in case_results)
    
    timeout_ms = 2000
    runtime_percent = max(0.0, min(100.0, (timeout_ms - avg_runtime_ms) / timeout_ms * 100.0))
    memory_threshold_mb = 256.0
    memory_percent = max(0.0, min(100.0, (memory_threshold_mb - max_memory_mb) / memory_threshold_mb * 100.0))
    final_percent = (accuracy_pct * 0.7) + (runtime_percent * 0.2) + (memory_percent * 0.1)
    
    # Generate suggestions (matching original logic)
    suggestions = []
    if accuracy_pct < 100.0:
        suggestions.append("Failing test cases indicate logic or edge-case errors. Check boundary conditions, off-by-one errors, and input parsing.")
    if any(r.get('status') == 'Timeout' for r in case_results):
        suggestions.append("One or more cases timed out — consider algorithmic improvements or faster I/O (e.g., use buffered reads).")
    if any(r.get('status') == 'Runtime Error' for r in case_results):
        suggestions.append("Runtime errors occurred. Inspect exception messages in case details and add defensive checks.")
    if avg_runtime_ms >= 0.8 * timeout_ms:
        suggestions.append("Average runtime is close to the timeout. Optimize inner loops or reduce overhead.")
    if max_memory_mb >= 0.9 * memory_threshold_mb:
        suggestions.append("High memory usage detected. Consider using streaming algorithms or smaller data structures.")
    if not suggestions:
        suggestions.append("No immediate issues detected. Consider adding more edge-case tests for confidence.")
    
    # Add previews for debugging (matching original logic)
    for r in case_results:
        expected = r.get('expected_output') if 'expected_output' in r else None
        if expected is None and 'expected' in r:
            expected = r['expected']
        if expected is not None:
            r['expected_preview'] = expected[:300]
        if 'output' in r and isinstance(r['output'], str):
            r['actual_preview'] = r['output'][:300]
    
    return {
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
            'weights': {'correctness': 0.7, 'runtime': 0.2, 'memory': 0.1}
        },
        "details": case_results
    }


class UsacoGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        self.benchmark_path = "."  # problems/ folder is in same directory
        # self.white_agent_url = os.getenv("WHITE_AGENT_URL")  # Configured at startup
        self.white_agent_url = "https://angeles-veterinary-tent-medieval.trycloudflare.com/to_agent/6979fd196d134e3a81644f6a7f229781"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Automatically evaluates all USACO problems.
        No user input required - just triggers the evaluation.
        """
        try:
            print("="*70)
            print("Green agent: Starting automatic USACO benchmark evaluation")
            print("="*70)
            
            # Check if white agent URL is configured
            if not self.white_agent_url:
                error_msg = "❌ WHITE_AGENT_URL not configured. Please set environment variable."
                print(error_msg)
                await event_queue.enqueue_event(new_agent_text_message(error_msg))
                return
            
            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"🚀 USACO Benchmark Evaluation Started\n"
                    f"White Agent: {self.white_agent_url}\n"
                    f"Problems: {', '.join(PROBLEMS_TO_TEST)}\n"
                    f"{'='*60}"
                )
            )
            
            # Verify problems directory exists
            problems_dir = os.path.join(self.benchmark_path, 'problems')
            if not os.path.isdir(problems_dir):
                error_msg = f"❌ Problems directory not found: {problems_dir}"
                print(error_msg)
                await event_queue.enqueue_event(new_agent_text_message(error_msg))
                return
            
            # Evaluate all problems automatically
            all_reports = []
            
            for i, problem in enumerate(PROBLEMS_TO_TEST, 1):
                print(f"\n{'='*70}")
                print(f"[{i}/{len(PROBLEMS_TO_TEST)}] Evaluating: {problem}")
                print(f"{'='*70}")
                
                await event_queue.enqueue_event(
                    new_agent_text_message(f"\n📝 [{i}/{len(PROBLEMS_TO_TEST)}] Evaluating: **{problem}**")
                )
                
                report = await evaluate_single_problem(problem, self.white_agent_url, self.benchmark_path)
                
                if "error" in report:
                    msg = f"❌ Error: {report['error']}"
                    print(msg)
                    await event_queue.enqueue_event(new_agent_text_message(msg))
                else:
                    success = report['analysis']['accuracy_pct'] == 100.0
                    emoji = "✅" if success else "❌"
                    
                    msg = f"""{emoji} **{report['problem']}**
• Passed: {report['summary']['passed_cases']}/{report['summary']['total_cases']} ({report['summary']['accuracy']})
• Avg Runtime: {report['summary']['avg_runtime_ms']} ms
• Max Memory: {report['summary']['max_memory_mb']} MB  
• Final Score: {report['analysis']['final_score_percent']}%
• Suggestions: {', '.join(report['analysis']['suggestions'])}
"""
                    print(f"\nReport:\n{msg}")
                    await event_queue.enqueue_event(new_agent_text_message(msg))
                
                all_reports.append(report)
            
            # Final summary
            total_problems = len(PROBLEMS_TO_TEST)
            successful = sum(1 for r in all_reports if "error" not in r and r.get('analysis', {}).get('accuracy_pct') == 100.0)
            
            summary_msg = f"""
{'='*60}
📊 **FINAL SUMMARY**
{'='*60}
Total Problems: {total_problems}
Successful: {successful}/{total_problems} ({successful/total_problems*100:.1f}%)
{'='*60}

✅ Evaluation Complete!
"""
            print(summary_msg)
            await event_queue.enqueue_event(new_agent_text_message(summary_msg))
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Critical Error:\n{str(e)}\n\n{traceback.format_exc()}"
            print(f"Green agent: {error_msg}")
            await event_queue.enqueue_event(new_agent_text_message(error_msg))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="usaco_green_agent", host="localhost", port=9001):
    print("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = os.getenv("AGENT_URL")
    agent_card_dict["url"] = url

    # Check problems directory
    problems_dir = "./problems"
    if os.path.isdir(problems_dir):
        problems = [d for d in os.listdir(problems_dir) if os.path.isdir(os.path.join(problems_dir, d))]
        print(f"✓ Problems directory found: {len(problems)} problems")
        print(f"  Problems: {', '.join(sorted(problems))}")
    else:
        print(f"⚠️  WARNING: Problems directory not found: {problems_dir}")
    
    print("="*70)

    request_handler = DefaultRequestHandler(
        agent_executor=UsacoGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    print(f"Starting server on {host}:{port}...")
    uvicorn.run(app.build(), host=host, port=port)
