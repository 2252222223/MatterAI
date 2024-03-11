from CEO.Base.CEO_Basetool import CEOBaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from CEO.Base.CEO_sk import sk
from DB.Base.Manager_agent import Departmental_Manager_agent, CEO_to_Manager_parse
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.Data_Proessing_Tools_integrated import Data_Proessing_tools_list

Expert_experience_path = "E:\pycharm\\MatterAI-0816-only-test\\DB\\DB_Manager_Tools\\Data_Collect_tools\\Data_Proessing_tools\\Expert_experience\\Expert_experience vector_store"
data_proess_agent = Departmental_Manager_agent(sk, Data_Proessing_tools_list, Expert_experience_path=Expert_experience_path,Expert_experience = False)


class Data_Proess_Schema(BaseModel):
    query: str = Field(description="Should be a dictionary. goal:what you expect this subordinates to accomplish. path:Addresses of tabular data that require further data processing.")


class Custom_Dtat_Proess_Tool(CEOBaseTool):
    name = "data_proess_agent"
    description = "This is one of your subordinates who is very good at taking the data collected and processing it further. "
    args_schema: Type[BaseModel] = Data_Proess_Schema

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print("Start Conver")
        print(query)
        query = str(query)
        query = CEO_to_Manager_parse(query)
        print("query:" + query)
        return data_proess_agent.run([query])

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")