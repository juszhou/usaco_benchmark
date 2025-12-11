# USACO BENCHMARK: Green Agent
a local benchmarking tool designed to evaluate Large Language Model (LLM) generated code against algorithmic problems from the USACO archives.

GreenAgent is optimized for AI input, providing metrics on **Runtime Efficiency**, **Memory Usage**, and **Functional Correctness**, while enforcing strict timeouts to handle unstable or non-terminating code.
## Features

* **Automated Evaluation:** Iterates through a folder of solution scripts and matches them to problem statements automatically.
* **Resource Monitoring:** Tracks peak memory usage (MB) and execution time (ms) for every test case.
* **Safety Sandbox:** Enforces strict timeouts (default: 2s) to prevent infinite loops from hanging the system.
* **Detailed Reporting:** Outputs comprehensive JSON reports for debugging and CSV summaries for data analysis.

## Prerequisites

* Python 3.8+
* `psutil` (for memory monitoring)


```bash
pip install psutil
```

## To run the green agent
Change the code in the **solutions** folder
and run the command:
```
python3 run_demo.py
```
