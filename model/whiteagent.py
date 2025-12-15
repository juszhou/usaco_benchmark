import os
import subprocess
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

class WhiteAgent:
    """
    A code generation agent for USACO problems using HuggingFace models.
    It generates code, tests it, and iterates to improve.
    """

    def __init__(self, benchmark_path: str = ".", max_iterations: int = 5, timeout: int = 2):
        load_dotenv()
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in .env")
        self.client = InferenceClient(token=self.api_key)
        self.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Code generation model
        self.benchmark_path = benchmark_path
        self.max_iterations = max_iterations
        self.timeout = timeout

    def generate_and_test_solution(self, problem_name: str):
        """
        Generates a solution using HuggingFace and tests it against test cases, iterating if necessary.

        Args:
            problem_name (str): The name of the problem.
        """
        problem_desc = os.path.join(self.benchmark_path, 'problems', problem_name, 'description.txt')
        if not os.path.isfile(problem_desc):
            print(f"Problem '{problem_name}' description not available.")
            return

        with open(problem_desc, 'r') as f:
            description = f.read()
        
        solution_path = f"solutions_agent/{problem_name}_generated.py"

        test_cases = self._find_test_cases(problem_name)
        if not test_cases:
            print(f"No test cases found for {problem_name}")
            return

        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Generate code using HuggingFace
            code = self._generate_code_with_hf(description)
            if not code:
                print("Failed to generate code.")
                continue

            # Save the code to a file
            os.makedirs("solutions_agent", exist_ok=True)
            with open(solution_path, 'w') as f:
                f.write(code)

            print(f"Generated code saved to {solution_path}")

            # Test the generated code
            passed, feedback = self._run_tests(solution_path, test_cases)
            if passed:
                print("All tests passed!")
                break
            else:
                description += f"\n\nPrevious attempt failed. Feedback: {feedback}\nPlease fix the code."
        else:
            print("Max iterations reached, solution not fully correct.")

    def _generate_code_with_hf(self, description: str) -> str:
        """
        Calls HuggingFace to generate Python code using chat_completion.
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
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=1000,
                temperature=0.1
            )
            code = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()
            return code
        except Exception as e:
            print(f"Error generating code: {e}")
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

    def _run_tests(self, script_path: str, test_cases: list) -> tuple:
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

            result = self._run_single_test(script_path, input_data)
            actual_output = result.get('output', '').strip()
            if result['status'] == 'Completed' and actual_output == expected_output:
                print(f"  Test {i}: PASS")
            else:
                print(f"  Test {i}: FAIL - Status: {result['status']}, Expected: '{expected_output}', Got: '{actual_output}'")
                all_passed = False
                feedback_parts.append(f"Test {i} failed: Input '{input_data.strip()}', Expected '{expected_output}', Got '{actual_output}' (Status: {result['status']})")
        feedback = "; ".join(feedback_parts)
        return all_passed, feedback

    def _run_single_test(self, script_path: str, input_data: str) -> dict:
        """Executes the solution against a single input."""
        result = {}
        try:
            process = subprocess.Popen(
                ['python3', script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            stdout, stderr = process.communicate(input=input_data, timeout=self.timeout)
            result['output'] = stdout
            if process.returncode != 0:
                result['status'] = 'Runtime Error'
                result['output'] = f"Error: {stderr}"
            else:
                result['status'] = 'Completed'
        except subprocess.TimeoutExpired:
            process.kill()
            result['status'] = 'Timeout'
            result['output'] = 'Timeout: Process exceeded time limit'
        except Exception as e:
            result['status'] = 'Execution Error'
            result['output'] = f'Execution Error: {str(e)}'
        return result

if __name__ == "__main__":
    agent = WhiteAgent()
    # Example
    agent.generate_and_test_solution("ride")