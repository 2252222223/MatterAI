import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool


def read_excel_data(f_path):
    if f_path.endswith(".csv"):
        data = pd.read_csv(f_path)
    elif f_path.endswith(".xls") or f_path.endswith(".xlsx"):
        data = pd.read_excel(f_path)
    # 获取列数
    num_columns = len(data.columns)
    # 获取每列的名称
    column_names = ', '.join(data.columns)
    # 获取第一行的内容
    first_row_values = ', '.join(str(value) for value in data.iloc[0])
    # 创建描述字符串
    description = f"The tabular table has {num_columns} columns, these column names is {column_names}. Examples of their corresponding values are {first_row_values}."
    return description


class View_Data_Schema(BaseModel):
    f_path: str = Field(description="Should be the address of the tabular data.")


class View_Data(BaseTool):
    name = "View_data"
    description = "Useful when you need an overview of tabular data."
    args_schema: Type[BaseModel] = View_Data_Schema

    def _run(self, f_path: str) -> str:

        return read_excel_data(f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")