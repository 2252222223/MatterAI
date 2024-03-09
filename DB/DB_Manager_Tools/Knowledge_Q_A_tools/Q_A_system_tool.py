from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Q_A_system import QA_Conversation
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from DB.Base.Available_Vector_Library import available_vectors


class DKSchema(BaseModel):
    query: str = Field(description="should be a professional issues related to {areas}.".format(areas=available_vectors))
    key_word: str = Field(description="Should be a field name, need to choose one from {areas}.".format(areas=available_vectors))


class DKquery(BaseTool):
    name = "Domain_knowledge_query"
    description = """Useful when you need answers to questions related to {areas}. """.format(areas=available_vectors)
    args_schema: Type[BaseModel] = DKSchema

    def _run(self, query: str, key_word: str) -> str:

        answer = QA_Conversation(query, key_word)

        return answer.get("summary_answer")

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")
