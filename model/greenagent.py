import os
import subprocess
import time
import threading
import psutil

class GreenAgent:
    """
    Acts as a hosting evaluator for programming tasks. It runs submitted code
    against a set of test cases, measures performance, and reports the results.
    """

    def __init__(self, benchmark_path: str, timeout_seconds: int = 2):
        """
        Initializes the agent.

        Args:
            benchmark_path (str): The root directory containing all USACO problems.
            timeout_seconds (int): The maximum time allowed for a single test case.
        """
        if not os.path.isdir(benchmark_path):
            raise FileNotFoundError(f"Benchmark path not found: {benchmark_path}")
        self.benchmark_path = benchmark_path
        self.timeout = timeout_seconds
        print(f"Green Agent initialized. Watching benchmark path") #: '{benchmark_path}'

    def _monitor_memory(self, process: subprocess.Popen, results_dict: dict):
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

    def evaluate_solution(self, problem_name: str, solution_script_path: str) -> dict:
        """
        Runs the full evaluation for a given problem and solution script.

        Args:
            problem_name (str): The name of the problem directory (e.g., 'ride').
            solution_script_path (str): The full path to the Python script to be tested.

        Returns:
            dict: A detailed report of the evaluation.
        """
        problem_dir = os.path.join(self.benchmark_path, 'problems', problem_name)
        if not os.path.isdir(problem_dir):
            return {"error": f"Problem '{problem_name}' not found."}
        if not os.path.isfile(solution_script_path):
            return {"error": f"Solution script not found: '{solution_script_path}'"}

        test_cases = self._find_test_cases(problem_dir)
        if not test_cases:
            return {"error": f"No test cases found in '{problem_dir}'."}

        print(f"\nEvaluating '{problem_name}' with solution '{os.path.basename(solution_script_path)}'...")
        
        case_results = []
        overall_passed = 0

        for i, (in_file, out_file) in enumerate(test_cases, 1):
            with open(in_file, 'r') as f_in:
                input_data = f_in.read()
            with open(out_file, 'r') as f_out:
                expected_output = f_out.read().strip()

            result = self._run_single_test(solution_script_path, input_data)
            
            if result['status'] == 'Completed':
                actual_output = result['output'].strip()
                if actual_output == expected_output:
                    result['verdict'] = 'Correct'
                    overall_passed += 1
                else:
                    result['verdict'] = 'Wrong Answer'
            else:
                result['verdict'] = result['status'] #Timeout Error

            print(f"  - Test Case #{i}: {result['verdict']} ({result['runtime_ms']:.2f} ms, {result['peak_memory_mb']:.2f} MB)")
            case_results.append(result)

        final_report = {
            "problem": problem_name,
            "solution": os.path.basename(solution_script_path),
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

    def _find_test_cases(self, problem_dir: str) -> list:
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

    def _run_single_test(self, script_path: str, input_data: str) -> dict:
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

            mem_thread = threading.Thread(target=self._monitor_memory, args=(process, result))
            mem_thread.start()

            stdout, stderr = process.communicate(input=input_data, timeout=self.timeout)
            
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
            result['runtime_ms'] = self.timeout * 1000
        
        except Exception as e:
            result['status'] = 'Execution Error'
            result['error_details'] = str(e)

        return result