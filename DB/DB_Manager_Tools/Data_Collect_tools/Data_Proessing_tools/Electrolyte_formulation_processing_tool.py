from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.Electrolyte_formulation_processing import ele_recipes_pro



class Electrolyte_composition_pro_Schema(BaseModel):
    f_path: str = Field(description="Should be the address of the tabular data.")
    col: str = Field(description="Should be a column name for a tabular form, related to a electrolyte composition.")


class Electrolyte_composition_pro(BaseTool):
    name = "electrolyte composition processing"
    description = "Useful when you need to convert the electrolyte composition column in a tabular table into a numeric format recognizable by machine learning."
    args_schema: Type[BaseModel] = Electrolyte_composition_pro_Schema

    def _run(self, f_path: str, col: str) -> str:

        return ele_recipes_pro(f_path,col)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")