from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Human_MatterAI_interaction import ContinuousDialogue
from DB.Base.Available_Vector_Library import available_vectors


def Q_A_Communication(key_word):
    dialogue = ContinuousDialogue(key_word)
    dialogue.start_conversation()
    return "Communication is over, and by communicating, the user has gained the corresponding domain knowledge."


class Q_A_Communication_Ahema(BaseModel):
    key_word: str = Field(description="Should be a field name, need to choose one from {areas}.".format(areas=available_vectors))


class DKCommunication(BaseTool):
    name = "Domain_knowledge_Communication"
    description = "Very useful when you need to gain knowledge from communication in the ares of {areas}.".format(areas = available_vectors)
    args_schema: Type[BaseModel] = Q_A_Communication_Ahema

    def _run(self, key_word: str) -> str:

        answer = Q_A_Communication(key_word)

        return answer

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")


