import os
import json
import time
from model.greenagent import GreenAgent 

BENCHMARK_ROOT = "."
#SOLUTIONS_DIR = "solutions"
PROBLEMS_TO_TEST = [
    "ride",
    "gift1",
    "friday"
]

def run_evaluation_demo():
    """
    Initializes the Green Agent and runs it on a set of problems,
    printing a report for each.
    """
    agent = GreenAgent(benchmark_path=BENCHMARK_ROOT, timeout_seconds=2)
    
    for i, problem in enumerate(PROBLEMS_TO_TEST):
        print("\n" + "="*50)
        print(f"TASK {i+1}/{len(PROBLEMS_TO_TEST)}: Evaluating '{problem}'")
        print("="*50)
        
        #solution_path = os.path.join(SOLUTIONS_DIR, solution_file)
        
            
        report = agent.evaluate_solution(problem)
        
        print("\n--- Final Report  ---")
        print(json.dumps(report, indent=2))
        
        if i < len(PROBLEMS_TO_TEST) - 1:
            time.sleep(2)

    print("\n" + "="*50)
    print("Demo finished. All tasks evaluated.")
    print("="*50)


if __name__ == "__main__":
    run_evaluation_demo()