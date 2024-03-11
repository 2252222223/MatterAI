from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.Autonomous_feature_engineering import Automated_Feature_Engineering_and_Screening



class Auto_Fe_Eng_Schema(BaseModel):
    f_path: str = Field(description="Should be the address of the tabular data.")


class Auto_Fe_Eng(BaseTool):
    name = "Automated Feature Engineering"
    description = "Useful when you need to use automated feature engineering to obtain optimal features."
    args_schema: Type[BaseModel] = Auto_Fe_Eng_Schema

    def _run(self, f_path: str) -> str:

        return Automated_Feature_Engineering_and_Screening(f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")