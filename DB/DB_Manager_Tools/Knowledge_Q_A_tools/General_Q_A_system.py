from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Chains.GPT4_answer_chain import General_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from DB.Base.Available_Vector_Library import available_vectors
memory = ConversationBufferWindowMemory(k=10)


def General_Q_A(query, memory=memory):
    history = memory.load_memory_variables({}).get("history")
    answer = General_chain({"history": history, "question": query})
    return answer


class GeneralQAScGeneral_Q_Ahema(BaseModel):
    query: str = Field(description="should be a questions within the field of materials science.")


class GeneralQAquery(BaseTool):
    name = "General_query"
    description = "Very useful when you need answers to materials science related questions, and this field is not in {available_vectors}.".format(available_vectors=available_vectors)
    args_schema: Type[BaseModel] = GeneralQAScGeneral_Q_Ahema

    def _run(self, query: str) -> str:

        answer = General_Q_A(query)

        return answer.get("text")

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

