# USACO BENCHMARK: Depoly on AgentBeats
Newest Assessment Link: https://v2.agentbeats.org/view/assessment/6f202fa0-f5d1-433c-a953-6521a5ecd1d2

# USACO BENCHMARK: Green Agent
a local benchmarking tool designed to evaluate Large Language Model (LLM) generated code against algorithmic problems from the USACO archives.

GreenAgent is optimized for AI input, providing metrics on **Runtime Efficiency**, **Memory Usage**, and **Functional Correctness**, while enforcing strict timeouts to handle unstable or non-terminating code.
## Features

* **Automated Evaluation:** Iterates through a folder of solution scripts and matches them to problem statements automatically.
* **Resource Monitoring:** Tracks peak memory usage (MB) and execution time (ms) for every test case.
* **Safety Sandbox:** Enforces strict timeouts (default: 2s) to prevent infinite loops from hanging the system.
* **Detailed Reporting:** Outputs comprehensive JSON reports for debugging and CSV summaries for data analysis.

## Prerequisites/Environment Setup

* Python 3.8+
* `psutil` (for memory monitoring)

```bash
pip install psutil
```
or you could run 
`uv sync`

## Usage for Agent Beats
Configure `.env` with `HUGGINGFACE_API_KEY=...`, then:

## To run the green agent demo
Change the code in the **solutions** folder
and run the command:
```
python3 run_demo.py
```
or to launch a specific problem:

```bash
python main.py launch --problem ride
python main.py launch --problem gift1
python main.py launch --problem friday
```
