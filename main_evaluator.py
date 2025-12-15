from model.greenagent import GreenAgent
import os
import json

BENCHMARK_ROOT = "."
#SOLUTIONS_DIR = "solutions"

if __name__ == "__main__":
    agent = GreenAgent(benchmark_path=BENCHMARK_ROOT, timeout_seconds=2)

    problem_to_test = 'ride'
    solution_to_test = 'ride_solution.py'

    #solution_path = os.path.join(SOLUTIONS_DIR, solution_to_test)

    report = agent.evaluate_solution(problem_to_test)

    print("\nFinal Report:")
    print(json.dumps(report, indent=2))