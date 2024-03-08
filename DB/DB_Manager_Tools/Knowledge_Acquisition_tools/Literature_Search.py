import os
import glob
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from CEO.Base.CEO_sk import sk
import shutil
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool

def obtain_pdf_name():
    # 设置文件夹路径
    folder_path = './pdf_test'
    # 遍历文件夹
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
    # 设置保存PDF名称的文本文件路径
    output_file = './pdf_name/all_pdf_name.txt'
    # 遍历文件夹并找到所有PDF文件
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
    # 将所有PDF文件的名称写入文本文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for pdf in pdf_files:
            file.write(f"{os.path.basename(pdf)}\n\n")


def pdf_name_embedding(txt_folder="./pdf_name/"):
    # 加载文件夹中的所有txt类型的文件
    loader = DirectoryLoader(txt_folder, glob='**/*.txt')
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()
    # 初始化加载器
    text_splitter = CharacterTextSplitter(
        separator="****",
        chunk_overlap=0,
        chunk_size=100,
        length_function=len,
    )
    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)
    print(documents)
    print(split_docs)
    # 初始化 openai 的 embeddings 对象
    embeddings = OpenAIEmbeddings(openai_api_key=sk)
    # 加载数据
    save_path = "./pdf_name/" + "vector_store"
    if not os.path.exists(save_path):  # 判断文件夹是否存在
        os.mkdir(save_path)  # 创建文件夹
    docsearch = Chroma(persist_directory=save_path, embedding_function=embeddings)
    # 添加新的文档和嵌入向量
    docsearch.add_documents(split_docs)
    # 保存更新后的向量数据库
    docsearch.persist()


def pdf_match(keyword:str):
    # 设置工作目录和目标目录
    print(os.path.abspath(__file__))
    source_folder = 'D:\OneDrive - mails.ucas.ac.cn\\Code\\未分类\\MatterAI-0816-only-test\\DB\\DB_Manager_Tools\\Knowledge_Acquisition_tools\\pdf_test'  # 源文件夹路径
    target_folder = 'D:\OneDrive - mails.ucas.ac.cn\\Code\\未分类\\MatterAI-0816-only-test\\DB\\DB_Manager_Tools\\Knowledge_Acquisition_tools\\' + keyword + '_path'  # 目标文件夹路径
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取源文件夹中所有PDF文件的名称
    pdf_files = [f for f in os.listdir(source_folder) if f.endswith('.pdf')]

    # 创建一个列表来存储包含关键词的文件名
    matched_files = [f for f in pdf_files if keyword in f]
    for file in matched_files:
        # 复制文件到目标文件夹
        shutil.copy(os.path.join(source_folder, file), target_folder)
    return target_folder


class PdfMatch_Schema(BaseModel):
    key_word: str = Field(description="Should be a keyword within a field.", example="Li-ion battery")


class PdfMatch(BaseTool):
    name = "pdf_match"
    description = "Useful when you need to collect literature in a specific field."
    args_schema: Type[BaseModel] = PdfMatch_Schema

    def _run(self, key_word: str) -> str:
        print(key_word)
        target_folder = pdf_match(key_word)
        response = "All the literature related to {key_word} has been found and saved in {target_folder} in pdf format.".format(key_word=key_word, target_folder=target_folder)
        return response

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")
















# obtain_pdf_name()
# pdf_name_embedding(txt_folder="./pdf_name/")

# key_world = "2020_ji_disordered_spinel_nature_energy"
# embeddings = OpenAIEmbeddings(openai_api_key=sk)
# docsearch = Chroma(persist_directory="./pdf_name/vector_store", embedding_function=embeddings)
# docs = docsearch.similarity_search(key_world, include_metadata=True)
# aaa = docsearch.similarity_search_with_score(key_world)
# # 打印每个文档的相似度分数
# for doc, score in aaa:
#     print(f"文档: {doc}, 相似度分数: {score}")

