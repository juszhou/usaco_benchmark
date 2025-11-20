"""White agent implementation - the target agent being tested."""

import re
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
from litellm import completion


dotenv.load_dotenv()


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="programming_task_solver",
        name="Programming Task Solver",
        description="Solves programming problems and generates code solutions",
        tags=["programming", "coding"],
        examples=[],
    )
    card = AgentCard(
        name="programming_agent",
        description="Agent that solves programming problems",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class GeneralWhiteAgentExecutor(AgentExecutor):
    def __init__(self):
        self.ctx_id_to_messages = {}

    @staticmethod
    def _ensure_solution_tags(content: str) -> str:
        """Guarantee that the outbound message includes <solution> tags."""
        if "<solution>" in content and "</solution>" in content:
            return content

        code_block = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if code_block:
            code = code_block.group(1).strip()
        else:
            code = content.strip()

        return f"<solution>\n{code}\n</solution>"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Parse the task
        user_input = context.get_user_input()
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []
        messages = self.ctx_id_to_messages[context.context_id]
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        response = completion(
            messages=messages,
            model="huggingface/meta-llama/Llama-4-Scout-17B-16E-Instruct",
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
            temperature=0.0,
        )
        next_message = response.choices[0].message.model_dump()  # type: ignore
        assistant_content = self._ensure_solution_tags(next_message["content"])
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )
        await event_queue.enqueue_event(
            new_agent_text_message(
                assistant_content, context_id=context.context_id
            )
        )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="general_white_agent", host="localhost", port=9002):
    print("Starting white agent...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=GeneralWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
