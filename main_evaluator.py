from model.greenagent import GreenAgent
import os
import json

BENCHMARK_ROOT = "."
SOLUTIONS_DIR = "solutions"

if __name__ == "__main__":
    agent = GreenAgent(benchmark_path=BENCHMARK_ROOT, timeout_seconds=2)

    problem_to_test = 'ride'
    solution_to_test = 'ride_solution.py'

    solution_path = os.path.join(SOLUTIONS_DIR, solution_to_test)

    report = agent.evaluate_solution(problem_to_test, solution_path)

    print("\nFinal Report Summary:")
    analysis = report.get('analysis', {})
    summary = report.get('summary', {})

    print(f" Problem: {report.get('problem')}")
    print(f" Solution: {report.get('solution')}")
    print(f" Passed: {summary.get('passed_cases')} / {summary.get('total_cases')} ({analysis.get('accuracy_pct', 'N/A')}%)")
    print(f" Avg runtime: {analysis.get('runtime_ms_avg', 'N/A')} ms")
    print(f" Max memory: {analysis.get('max_memory_mb', 'N/A')} MB")
    print(f" Final score: {analysis.get('final_score_percent', 'N/A')}%")

    print("\nSuggestions:")
    for s in analysis.get('suggestions', []):
        print(f" - {s}")

    print("\nFull JSON report:\n")
    print(json.dumps(report, indent=2))