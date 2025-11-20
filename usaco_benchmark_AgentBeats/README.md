# USACO Benchmark

USACO Green Agent - AgentBeats compatible benchmark for evaluating programming problem-solving capabilities using USACO (USA Computing Olympiad) problems.

## Overview

This project implements a **green agent** (hosting agent) compatible with the AgentBeats platform. The green agent evaluates white agents (participant agents) by testing their ability to solve USACO programming problems.

### What is a Green Agent?

A green agent is a special hosting agent that:
- Defines the assessment type and specific tasks
- Manages a dataset of test tasks (USACO problems)
- Defines the testing process and workflow
- Evaluates participant agents (white agents)
- Provides necessary tools and environment for testing

## Project Structure

```
.
├── src/
│   ├── green_agent/       # Assessment manager agent
│   │   ├── agent.py       # Green agent implementation
│   │   └── usaco_green_agent.toml  # Agent configuration
│   ├── white_agent/       # Example target agent being tested
│   │   └── agent.py       # White agent implementation
│   ├── my_util/           # Utility functions
│   │   ├── parse_tags.py  # Tag parsing helper
│   │   └── my_a2a.py      # A2A protocol helpers
│   └── launcher.py        # Evaluation coordinator
├── problems/              # USACO problem test cases
│   ├── ride/
│   ├── friday/
│   └── gift1/
├── solutions/             # Example solutions
├── model/                 # Original evaluation logic
├── main.py                # CLI entry point
└── pyproject.toml         # Project dependencies
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management, but you can also use pip:

### Using uv (recommended)

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Using pip

```bash
pip install -e .
```

## Configuration

Before running, configure your environment:

1. Create a `.env` file in the project root:
```bash
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

## Usage

### Complete End-to-End Evaluation

Launch a complete evaluation with both green and white agents:

```bash
# Using uv
uv run python main.py launch

# Evaluate a specific problem
uv run python main.py launch --problem ride --timeout 2

# Or with pip
python main.py launch
```

This will:
1. Start the green agent (assessment manager) on port 9001
2. Start the white agent (agent being tested) on port 9002
3. Send the USACO problem to the white agent
4. Evaluate the white agent's solution against test cases
5. Display results and metrics

### Run Agents Separately

#### Start Green Agent Only

```bash
uv run python main.py green
```

#### Start White Agent Only

```bash
uv run python main.py white
```

## How It Works

### Green Agent Workflow

1. **Receives evaluation request** via A2A protocol containing:
   - White agent URL
   - Problem configuration (problem name, timeout)

2. **Sends problem description** to the white agent with:
   - Problem statement
   - Sample input/output
   - Expected output format (Python solution in `<solution>...</solution>` tags)

3. **Receives solution** from the white agent

4. **Evaluates solution** by:
   - Saving the code to a temporary file
   - Running it against all test cases
   - Measuring runtime and memory usage
   - Comparing outputs

5. **Returns results** including:
   - Success/failure status
   - Accuracy percentage
   - Test case details
   - Performance metrics

### White Agent

The included white agent is a simple LLM-based agent that:
- Receives problem descriptions
- Uses Meta Llama-4-Scout-17B-16E-Instruct to generate solutions
- Returns Python code in the required format

You can replace this with your own agent implementation for testing.

## A2A Protocol Compliance

This green agent follows the A2A (Agent-to-Agent) protocol:

- **Agent Card**: Defined in `usaco_green_agent.toml`
- **Skills**: Declares "host_assess_usaco" skill for USACO problem evaluation
- **Communication**: Uses standardized message format with A2A SDK
- **Task Format**: Expects tasks in XML-tagged format:
  ```xml
  <white_agent_url>http://localhost:9002/</white_agent_url>
  <problem_config>{"problem_name": "ride", "timeout_seconds": 2}</problem_config>
  ```

## Available Problems

The benchmark includes several USACO problems:
- `ride` - The Name Game
- `friday` - Friday the Thirteenth
- `gift1` - Greedy Gift Givers

Each problem has:
- Input/output test cases (`.in`, `.out` files)
- Example solutions in the `solutions/` directory

## Extending the Benchmark

### Adding New Problems

1. Create a new directory in `problems/`:
```bash
mkdir problems/newproblem
```

2. Add test cases:
```
problems/newproblem/
  ├── newproblem.in    # Input
  └── newproblem.out   # Expected output
```

3. (Optional) Add problem description:
```
problems/newproblem/newproblem.txt
```

### Custom White Agent

Replace the white agent with your own implementation:

```python
from a2a.server.agent_execution import AgentExecutor

class MyAgentExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        # Your agent logic here
        pass
```

## Testing Locally

To verify the setup works:

```bash
# Run complete evaluation on the 'ride' problem
uv run python main.py launch --problem ride

# Expected output:
# - Both agents start successfully
# - Problem is sent to white agent
# - Solution is evaluated
# - Results show pass/fail for each test case
```

## Dependencies

- `a2a-sdk[http-server]>=0.3.8` - A2A protocol implementation
- `typer>=0.19.2` - CLI framework
- `uvicorn>=0.37.0` - ASGI server
- `psutil>=5.9.0` - Process monitoring
- `litellm>=1.0.0` - LLM integration
- `dotenv>=0.9.9` - Environment configuration

## License

This project is for educational and research purposes.

## Acknowledgments

- USACO (USA Computing Olympiad) for the problem set
- AgentBeats platform for the A2A protocol specification
