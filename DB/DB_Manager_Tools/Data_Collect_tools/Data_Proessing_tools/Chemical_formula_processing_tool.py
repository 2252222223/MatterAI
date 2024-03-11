from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.Chemical_formula_processing import split_chemical_formula


class split_chemical_formula_Schema(BaseModel):
    f_path: str = Field(description="Should be the address of the tabular data.")
    col: str = Field(description="Should be a column name for a tabular form, related to a chemical formula.")


class Chemical_formula_pro(BaseTool):
    name = "chemical formula processing"
    description = "Useful when you need to convert the chemical formula column in a tabular table into a numeric format recognizable by machine learning."
    args_schema: Type[BaseModel] = split_chemical_formula_Schema

    def _run(self, f_path: str, col: str) -> str:

        return split_chemical_formula(f_path,col)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")
