"""CLI entry point for USACO Agent Evaluation System."""

import typer
import asyncio
from pydantic_settings import BaseSettings

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation, launch_remote_evaluation


class UsacoSettings(BaseSettings):
    role: str = "unspecified"
    host: str = "127.0.0.1"
    agent_port: int = 8011


app = typer.Typer(help="USACO Agent Evaluation System - Agentified code evaluation framework")


@app.command()
def green():
    """Start the green agent (evaluation manager)."""
    start_green_agent()


@app.command()
def white():
    """Start the white agent (code generation agent)."""
    start_white_agent()


@app.command()
def run():
    """Run agent based on role setting (green/white)."""
    settings = UsacoSettings()
    if settings.role == "green":
        start_green_agent(host=settings.host, port=settings.agent_port)
    elif settings.role == "white":
        start_white_agent(host=settings.host, port=settings.agent_port)
    else:
        raise ValueError(f"Unknown role: {settings.role}")
    return


@app.command()
def launch():
    """Launch the complete evaluation workflow."""
    asyncio.run(launch_evaluation())


@app.command()
def launch_remote(green_url: str, white_url: str):
    """Launch evaluation with remote agents."""
    asyncio.run(launch_remote_evaluation(green_url, white_url))


if __name__ == "__main__":
    app()
