
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data


def data_clean(file_path: str):
    data = read_excel_data(file_path)
    return "Data processing has been completed and the processed dataset is saved in ./Li_excess_dataset.xlsx."


class Data_clean_PreSchema(BaseModel):

    f_path: str = Field(description="Should be the address of a dataset that needs to be processed.")


class Data_clean(BaseTool):
    name = "Data_clean"
    description = "Useful when you need to perform further processing on a collected dataset."
    args_schema: Type[BaseModel] = Data_clean_PreSchema

    def _run(self, f_path: str) -> str:

        return data_clean(f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

