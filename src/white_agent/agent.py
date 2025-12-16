"""White agent implementation - code generation agent for USACO problems."""

import uvicorn
import dotenv
import os
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from huggingface_hub import InferenceClient

dotenv.load_dotenv()


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="code_generation",
        name="Code Generation",
        description="Generates Python solutions for USACO programming problems",
        tags=["coding", "usaco"],
        examples=[],
    )
    card = AgentCard(
        name="usaco_code_generator",
        description="Code generation agent for USACO problems using HuggingFace models",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class UsacoWhiteAgentExecutor(AgentExecutor):
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in .env")
        self.client = InferenceClient(token=self.api_key)
        self.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        
        print("White agent: Received code generation request...")
        
        # Generate code using HuggingFace
        code = await self._generate_code(user_input)
        
        if code:
            response_message = f"<code>\n{code}\n</code>"
            print(f"White agent: Code generated successfully.")
        else:
            response_message = "<code>\n# Failed to generate code\npass\n</code>"
            print("White agent: Failed to generate code.")
        
        await event_queue.enqueue_event(
            new_agent_text_message(response_message, context_id=context.context_id)
        )

    async def _generate_code(self, description: str) -> str:
        """Generate Python code using HuggingFace API."""
        system_prompt = "You are a helpful assistant that writes Python code for programming problems."
        user_prompt = f"""Write a complete Python program that solves the following USACO problem. The program should read from stdin and write to stdout. Only provide the Python code, no explanations or markdown formatting.

{description}
"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=1000,
                temperature=0.1
            )
            code = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()
            
            return code
        except Exception as e:
            print(f"Error generating code: {e}")
            return None

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="usaco_white_agent", host="localhost", port=8011):
    print("Starting white agent...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=UsacoWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
