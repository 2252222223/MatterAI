from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from CEO.Base.CEO_sk import sk
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool


python = PythonREPLTool()
py = Tool(
    name="Python shell",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python.run,
)


python_agent = create_python_agent(
    llm=ChatOpenAI(model_name="gpt-4-0613", temperature=0, max_tokens=1000, openai_api_key=sk),
    tool=py,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


class Python_PreSchema(BaseModel):

    requirements: str = Field(description="Should be specific task details, including data or tables, specific operational requirements for the data or tables.")
    goal_path: str = Field(description="The address where the form will be saved after the task is completed. such as ./single_data.csv")


class Py_shell(BaseTool):
    name = "Excel_python"
    description = "Very useful when modifying or creating tables. For example, converting a textual description of a sample in a task into a table for direct recall by other tools."
    args_schema: Type[BaseModel] = Python_PreSchema

    def _run(self, requirements: str, goal_path: str) -> str:
        goal = "your goal is " + requirements + ", then  you need to save the new form to " + goal_path + "."
        return python_agent.run(goal)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")



