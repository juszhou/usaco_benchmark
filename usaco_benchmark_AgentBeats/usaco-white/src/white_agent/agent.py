"""White agent implementation - USACO code generation agent."""

import uvicorn
import dotenv
import os
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from huggingface_hub import InferenceClient

dotenv.load_dotenv()

# All problems to process automatically
PROBLEMS_TO_PROCESS = ["ride", "gift1", "friday"]


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="code_generation",
        name="Code Generation",
        description="Generates Python solutions for USACO programming problems",
        tags=["coding", "usaco"],
        examples=[],
    )
    card = AgentCard(
        name="usaco_code_generator",
        description="Code generation agent for USACO problems using HuggingFace models",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class UsacoWhiteAgentExecutor(AgentExecutor):
    def __init__(self, benchmark_path: str = ".", max_iterations: int = 3, timeout: int = 2):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment")
        self.client = InferenceClient(token=self.api_key)
        self.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.benchmark_path = benchmark_path
        self.max_iterations = max_iterations
        self.timeout = timeout
        # Thread pool for running synchronous HuggingFace API calls
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Automatically processes all problems: ride, gift1, friday.
        No user input required - processes them sequentially.
        """
        try:
            print("="*70)
            print("White agent: Starting automatic code generation for all problems")
            print("="*70)
            
            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"🚀 White Agent: Starting Code Generation\n"
                    f"Problems: {', '.join(PROBLEMS_TO_PROCESS)}\n"
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
            
            # Process each problem sequentially
            all_results = []
            
            for i, problem_name in enumerate(PROBLEMS_TO_PROCESS, 1):
                print(f"\n{'='*70}")
                print(f"[{i}/{len(PROBLEMS_TO_PROCESS)}] Processing: {problem_name}")
                print(f"{'='*70}")
                
                await event_queue.enqueue_event(
                    new_agent_text_message(f"\n📝 [{i}/{len(PROBLEMS_TO_PROCESS)}] Processing: **{problem_name}**")
                )
                
                # Check if problem directory exists
                problem_dir = os.path.join(self.benchmark_path, 'problems', problem_name)
                if not os.path.isdir(problem_dir):
                    error_msg = f"❌ Problem directory not found: {problem_dir}"
                    print(error_msg)
                    await event_queue.enqueue_event(new_agent_text_message(error_msg))
                    all_results.append({"problem": problem_name, "error": error_msg})
                    continue
                
                # Generate and test solution
                code, feedback = await self._generate_and_test_solution(problem_name, event_queue)
                
                if code:
                    success = "All tests passed!" in feedback
                    emoji = "✅" if success else "⚠️"
                    
                    result_msg = f"""{emoji} **{problem_name}**
• Code generated: ✓
• Status: {feedback}
"""
                    print(f"\nResult:\n{result_msg}")
                    await event_queue.enqueue_event(new_agent_text_message(result_msg))
                    all_results.append({"problem": problem_name, "success": success, "code": code, "feedback": feedback})
                else:
                    error_msg = f"❌ Failed to generate code for {problem_name}"
                    print(error_msg)
                    await event_queue.enqueue_event(new_agent_text_message(error_msg))
                    all_results.append({"problem": problem_name, "error": "Failed to generate code"})
            
            # Final summary
            total_problems = len(PROBLEMS_TO_PROCESS)
            successful = sum(1 for r in all_results if r.get('success', False))
            
            summary_msg = f"""
{'='*60}
📊 **FINAL SUMMARY**
{'='*60}
Total Problems: {total_problems}
Successful: {successful}/{total_problems} ({successful/total_problems*100:.1f}%)
{'='*60}

✅ Code Generation Complete!
"""
            print(summary_msg)
            await event_queue.enqueue_event(new_agent_text_message(summary_msg))
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Critical Error:\n{str(e)}\n\n{traceback.format_exc()}"
            print(f"White agent: {error_msg}")
            await event_queue.enqueue_event(
                new_agent_text_message(error_msg, context_id=context.context_id)
            )

    async def _generate_and_test_solution(self, problem_name: str, event_queue: EventQueue):
        """
        Generates a solution using HuggingFace and tests it against test cases, iterating if necessary.
        Returns (code, feedback) tuple.
        """
        problem_dir = os.path.join(self.benchmark_path, 'problems', problem_name)
        desc_path = os.path.join(problem_dir, 'description.txt')
        
        if not os.path.isfile(desc_path):
            return None, f"Problem '{problem_name}' description not available."
        
        with open(desc_path, 'r') as f:
            description = f.read()
        
        solution_dir = os.path.join(self.benchmark_path, 'solutions_agent')
        os.makedirs(solution_dir, exist_ok=True)
        solution_path = os.path.join(solution_dir, f"{problem_name}_generated.py")
        
        test_cases = self._find_test_cases(problem_name)
        if not test_cases:
            return None, f"No test cases found for {problem_name}"
        
        for iteration in range(self.max_iterations):
            await event_queue.enqueue_event(
                new_agent_text_message(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")
            )
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate code using HuggingFace
            code = await self._generate_code_with_hf(description)
            if not code:
                await event_queue.enqueue_event(
                    new_agent_text_message("❌ Failed to generate code.")
                )
                continue
            
            # Save the code to a file
            with open(solution_path, 'w') as f:
                f.write(code)
            
            print(f"Generated code saved to {solution_path}")
            
            # Test the generated code
            passed, feedback = await self._run_tests(solution_path, test_cases, event_queue)
            if passed:
                await event_queue.enqueue_event(
                    new_agent_text_message("✅ All tests passed!")
                )
                print("All tests passed!")
                return code, "All tests passed!"
            else:
                description += f"\n\nPrevious attempt failed. Feedback: {feedback}\nPlease fix the code."
                await event_queue.enqueue_event(
                    new_agent_text_message(f"⚠️ Tests failed: {feedback}")
                )
        
        await event_queue.enqueue_event(
            new_agent_text_message(f"⚠️ Max iterations ({self.max_iterations}) reached, solution may not be fully correct.")
        )
        print("Max iterations reached, solution not fully correct.")
        return code, f"Max iterations reached. Last feedback: {feedback}"

    async def _generate_code_with_hf(self, description: str) -> str:
        """
        Calls HuggingFace to generate Python code using chat_completion.
        Uses thread pool executor to avoid blocking the event loop.
        """
        system_prompt = "You are a helpful assistant that writes Python code for programming problems."
        user_prompt = f"""Write a complete Python program that solves the following USACO problem. The program should read from stdin and write to stdout. Only provide the Python code, no explanations.

Problem Description:
{description}
"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Run synchronous API call in thread pool with timeout
            loop = asyncio.get_event_loop()
            print(f"  [API] Calling HuggingFace API...")
            start_time = asyncio.get_event_loop().time()
            
            def _call_api():
                return self.client.chat_completion(
                    messages=messages,
                    model=self.model_id,
                    max_tokens=1000,
                    temperature=0.1
                )
            
            response = await asyncio.wait_for(
                loop.run_in_executor(self.executor, _call_api),
                timeout=120.0  # 120 second timeout for API call
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"  [API] Response received in {elapsed:.2f}s")
            
            code = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()
            return code
        except asyncio.TimeoutError:
            print(f"  [API] Error: API timeout after 120 seconds")
            return None
        except Exception as e:
            print(f"  [API] Error generating code: {e}")
            return None

    def _find_test_cases(self, problem_name: str) -> list:
        """Finds pairs of input/output files."""
        problem_dir = os.path.join(self.benchmark_path, 'problems', problem_name)
        if not os.path.isdir(problem_dir):
            return []
        files = os.listdir(problem_dir)
        in_files = sorted([f for f in files if f.endswith('.in')])
        test_cases = []
        for in_file in in_files:
            base_name = in_file.rsplit('.', 1)[0]
            out_file = f"{base_name}.out"
            if out_file in files:
                test_cases.append((os.path.join(problem_dir, in_file), os.path.join(problem_dir, out_file)))
        return test_cases

    async def _run_tests(self, script_path: str, test_cases: list, event_queue: EventQueue) -> tuple:
        """
        Runs the script against test cases and returns if all passed and feedback.
        """
        all_passed = True
        feedback_parts = []
        for i, (in_file, out_file) in enumerate(test_cases, 1):
            with open(in_file, 'r') as f_in:
                input_data = f_in.read()
            with open(out_file, 'r') as f_out:
                expected_output = f_out.read().strip()
            
            result = await self._run_single_test(script_path, input_data)
            actual_output = result.get('output', '').strip()
            if result['status'] == 'Completed' and actual_output == expected_output:
                print(f"  Test {i}: PASS")
            else:
                print(f"  Test {i}: FAIL - Status: {result['status']}, Expected: '{expected_output}', Got: '{actual_output}'")
                all_passed = False
                feedback_parts.append(f"Test {i} failed: Input '{input_data.strip()}', Expected '{expected_output}', Got '{actual_output}' (Status: {result['status']})")
        feedback = "; ".join(feedback_parts)
        return all_passed, feedback

    async def _run_single_test(self, script_path: str, input_data: str) -> dict:
        """Executes the solution against a single input using async subprocess."""
        result = {}
        try:
            # Use async subprocess to avoid blocking
            process = await asyncio.create_subprocess_exec(
                'python3', script_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data.encode('utf-8')),
                    timeout=self.timeout
                )
                result['output'] = stdout.decode('utf-8')
                if process.returncode != 0:
                    result['status'] = 'Runtime Error'
                    result['output'] = f"Error: {stderr.decode('utf-8')}"
                else:
                    result['status'] = 'Completed'
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                result['status'] = 'Timeout'
                result['output'] = 'Timeout: Process exceeded time limit'
        except Exception as e:
            result['status'] = 'Execution Error'
            result['output'] = f'Execution Error: {str(e)}'
        return result

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="usaco_white_agent", host="localhost", port=8011, benchmark_path=".", max_iterations=5, timeout=2):
    print("Starting white agent...")
    url = os.getenv("AGENT_URL")
    card = prepare_white_agent_card(url)
    
    request_handler = DefaultRequestHandler(
        agent_executor=UsacoWhiteAgentExecutor(benchmark_path=benchmark_path, max_iterations=max_iterations, timeout=timeout),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    print(f"Starting server on {host}:{port}")
    uvicorn.run(app.build(), host=host, port=port)
