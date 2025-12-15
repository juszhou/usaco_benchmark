"""CLI entry point for USACO benchmark."""

import typer
import asyncio

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation

app = typer.Typer(help="USACO Benchmark - AgentBeats compatible green agent")


@app.command()
def green():
    """Start the green agent (assessment manager)."""
    start_green_agent()


@app.command()
def white():
    """Start the white agent (target being tested)."""
    start_white_agent()


@app.command()
def launch(
    problem: str = typer.Option("ride", help="Problem name to evaluate"),
    timeout: int = typer.Option(2, help="Timeout in seconds for each test case")
):
    """Launch the complete evaluation workflow."""
    asyncio.run(launch_evaluation(problem_name=problem, timeout_seconds=timeout))


if __name__ == "__main__":
    app()
