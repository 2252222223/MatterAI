from CEO.Base.CEO_sk import sk
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data


def qury_excel(query, df):
    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0, max_tokens=1000, openai_api_key=sk)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    response = agent.run(query)
    return response


def query_tool(query:str, file_path:str):
    df = read_excel_data(file_path)
    response = qury_excel(query,df)
    return response


class Query_CSV_PreSchema(BaseModel):

    query: str = Field(description="It should be a target indicating the query or operation to be performed on the table.")
    f_path: str = Field(description="This should be the address of the table to be queried or manipulated.")


class Query_CSV(BaseTool):
    name = "Query_CSV"
    description = "Useful when you need to query the contents of a table or extract information from a table."
    args_schema: Type[BaseModel] = Query_CSV_PreSchema

    def _run(self, query: str, f_path: str) -> str:

        return query_tool(query, f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")