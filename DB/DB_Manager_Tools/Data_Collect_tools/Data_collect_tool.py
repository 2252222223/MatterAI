import os
import json
import numpy as np
import pandas as pd
from DB.DB_Manager_Tools.Data_Collect_tools.Gnerate_query import generate_query
from DB.DB_Manager_Tools.Data_Collect_tools.Feature_jason import generate_attributes
from DB.DB_Manager_Tools.Data_Collect_tools.Find_materials import find_materials
from DB.DB_Manager_Tools.Data_Collect_tools.Match_text import match_text
from DB.DB_Manager_Tools.Data_Collect_tools.Extract_data import creat_extract_chain, collect_paper_data
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.Data_Collect_tools.Self_query_completion import fill_data_None,fill_abbreviations,get_full_paper


#读取当前文件夹所有pdf
def all_file_name(file_dir, source_type=".txt"):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == source_type:
                z = file.split(".")
                L.append(file_dir + "/" + os.path.splitext(file)[0] + source_type)
                        #     print(L)
    return L


def collect_data(query: str, file_path: str):
    L = all_file_name(file_path, source_type=".txt")
    attributes, materials_feature = generate_attributes(query)
    new_dataset = pd.DataFrame(columns=list(materials_feature.keys()))
    extract_chain = creat_extract_chain(attributes)
    for i in range(len(L)):
        path = L[i]
        simi_docs, split_docs = match_text(path,query)
        dataset = pd.DataFrame(columns=materials_feature.keys())  # 新建一个表格，用于储存数据
        new_dataset2 =collect_paper_data(dataset,extract_chain,simi_docs)
        paper = get_full_paper(split_docs)
        new_dataset2 = fill_data_None(new_dataset2, paper)                                        #self_query completion
        new_dataset2 = fill_abbreviations(new_dataset2, paper)
        new_dataset = pd.concat((new_dataset, new_dataset2), axis=0, ignore_index=True)
    new_dataset.to_csv("./data_collect.csv", index=False)
    return "Data collection from the literature has been completed and the dataset is saved at ./data_collect.csv"


class Data_collect_PreSchema(BaseModel):

    query: str = Field(description="Should be a query that contains the attributes of the data you want to collect.")
    f_path: str = Field(description="Should be the address of the literature in txt format.")


class Data_Collect(BaseTool):
    name = "Data_collect"
    description = "Very useful when you need to collect data from the literature."
    args_schema: Type[BaseModel] = Data_collect_PreSchema

    def _run(self, query: str, f_path: str) -> str:

        return collect_data(query, f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

