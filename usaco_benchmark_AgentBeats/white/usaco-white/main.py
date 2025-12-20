# """CLI entry point for USACO Agent Evaluation System."""

# import typer
# import asyncio
# from pydantic_settings import BaseSettings

# from src.green_agent import start_green_agent
# from src.white_agent import start_white_agent
# from src.launcher import launch_evaluation, launch_remote_evaluation


# class UsacoSettings(BaseSettings):
#     role: str = "unspecified"
#     host: str = "127.0.0.1"
#     agent_port: int = 8011


# app = typer.Typer(help="USACO Agent Evaluation System - Agentified code evaluation framework")


# @app.command()
# def green():
#     """Start the green agent (evaluation manager)."""
#     start_green_agent()


# @app.command()
# def white():
#     """Start the white agent (code generation agent)."""
#     start_white_agent()


# @app.command()
# def run():
#     """Run agent based on role setting (green/white)."""
#     settings = UsacoSettings()
#     if settings.role == "green":
#         start_green_agent(host=settings.host, port=settings.agent_port)
#         # start_green_agent(host="roots-frank-stanford-filme.trycloudflare.com", port="8010")
#     elif settings.role == "white":
#         start_white_agent(host=settings.host, port=settings.agent_port)
#         # start_white_agent(host="ebooks-written-frequent-cove.trycloudflare.com", port="8011")
#     else:
#         raise ValueError(f"Unknown role: {settings.role}")
#     return


# @app.command()
# def launch():
#     """Launch the complete evaluation workflow."""
#     asyncio.run(launch_evaluation())


# @app.command()
# def launch_remote(green_url: str, white_url: str):
#     """Launch evaluation with remote agents."""
#     asyncio.run(launch_remote_evaluation(green_url, white_url))


# if __name__ == "__main__":
#     app()
"""CLI entry point for USACO White Agent (Code Generator)."""

import os
from pydantic_settings import BaseSettings
from src.white_agent import start_white_agent


class UsacoSettings(BaseSettings):
    host: str = "0.0.0.0"
    agent_port: int = 8011
    benchmark_path: str = "."
    max_iterations: int = 5
    timeout: int = 2
    
    # Cloudflare tunnel and HTTPS support
    https_enabled: bool = False
    cloudrun_host: str | None = None
    
    model_config = {"env_file": ".env", "extra": "ignore"}


if __name__ == "__main__":
    settings = UsacoSettings()
    start_white_agent(
        host=settings.host, 
        port=settings.agent_port,
        benchmark_path=settings.benchmark_path,
        max_iterations=settings.max_iterations,
        timeout=settings.timeout
    )