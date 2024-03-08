#!/usr/bin/env python
# coding: utf-8


from DB.DB_Manager_Tools.Knowledge_Acquisition_tools.Pdf_to_txt.pdf_Convert import Convert
from langchain.tools import BaseTool
import os


class PdfConvert(BaseTool):
    name = "pdf_to_txt"
    description = """Useful when you need to convert literature in pdf format to txt format. 
    Input: a dictionary, which contains two parameters source_path and key_world, which source_path represent the pdf document address, key_world represent the domain keywords of these documents.
    Here, this is an example:{"source_path":"./polymer_processing","key_world":"polymer processing"}"""

    def _run(self, source_path: str, key_world: str) -> str:
        if os.path.isdir(source_path) is False:
            return f"{source_path} does not exist!"

        goal_file = "D:\pycharm\\MatterAI-0816\\DB\\DB_Manager_Tools\\Knowledge_Acquisition_tools\\" + key_world + "_txt"
        if not os.path.isdir(goal_file):
            # 创建文件夹
            os.mkdir(goal_file)
        Convert(source_path, goal_file)
        response = """All pdf conversions have been completed, the corresponding txt format files are saved in 
        {goal_file}.Note that if the task is for pdf to txt format conversion only, you must report the address of the 
        converted txt file to the CEO.""".format(goal_file=goal_file)
        return response

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")
