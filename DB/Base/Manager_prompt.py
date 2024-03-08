from typing import Any, Callable, List
# from langchain_experimental.autonomous_agents.autogpt.prompt_generator import get_prompt
from DB.Base.prompt import get_prompt
from langchain.tools.base import BaseTool
from CEO.Base.CEO_prompt import CEO_GPT_Prompt


class Manager_Prompt(CEO_GPT_Prompt):
    ai_name: str
    ai_role: str
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    def construct_full_prompt(self, goals: List[str]) -> str:
        prompt_start = (
            "Your decisions must always be made independently "
            "without seeking user assistance.\n"
            "Play to your strengths as an LLM and pursue simple "
            "strategies with no legal complications.\n"
            "If you have completed your task, make sure to "
            'use the "finish" command, and YOU MUST report the detailed observations result to the CEO.'
        )

        # Construct full prompt
        full_prompt = (
            f"You are a department manager, your role is to accomplish the goal assigned by the CEO. \n{prompt_start}\n\nGOALS:\n\n"
        )
        for i, goal in enumerate(goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{get_prompt(self.tools)}"

        return full_prompt