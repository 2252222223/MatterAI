#!/usr/bin/env python
# coding: utf-8
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from CEO.Base.CEO_sk import sk
from langchain.tools import BaseTool
import os


def tex_to_embedding(txt_folder, key_world):
    # 加载文件夹中的所有txt类型的文件
    loader = DirectoryLoader(txt_folder, glob='**/*.txt')
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()
    # 初始化加载器
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )
    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)
    # 初始化 openai 的 embeddings 对象
    embeddings = OpenAIEmbeddings(openai_api_key=sk)
    # 加载数据
    save_path = "D:\pycharm\\MatterAI-0816\\DB\\DB_Manager_Tools\\Knowledge_Acquisition_tools\\Domain Vector Library\\" + key_world + " vector_store"
    if not os.path.exists(save_path):  # 判断文件夹是否存在
        os.mkdir(save_path)  # 创建文件夹
    docsearch = Chroma(persist_directory=save_path, embedding_function=embeddings)
    for i in range(len(split_docs) // 100 + 1):
        print(i)
        # 添加新的文档和嵌入向量
        docsearch.add_documents(split_docs[i * 100:(i + 1) * 100])
        # 保存更新后的向量数据库
        docsearch.persist()
    response = "All the documents in txt format have been converted to word embeddings and saved in the vector repository at {file}, Now that I am on the cutting edge of the field, you can discuss with me on issues related to {area}.".format(
        file=save_path, area=key_world)
    return response


class Txtembedding(BaseTool):
    name = "txt_to_embedding"
    description = """Useful when you need to learn domain knowledge from text-formatted literature, its duty is to convert literature in text format into vector storage for retrieval by the model when answering domain questions. 
    Input: a dictionary, which contains two parameters txt_folder and key_world, which source_path represent the txt document address, key_world represent the domain keywords of these documents.
    Here, this is an example:{"txt_folder":"../test//polymer_processing_txt","key_world":"polymer processing"}"""

    def _run(self, txt_folder: str, key_world: str) -> str:
        response = tex_to_embedding(txt_folder, key_world)
        return response

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")
