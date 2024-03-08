import os
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
import ast
# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from DB.Base.Manager_Auto_gpt import Manager_GPT


def memory_store():
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    return vectorstore


def Departmental_Manager_agent(sk: str, tools_list: list, Expert_experience_path:str,Expert_experience: bool = False) -> AutoGPT:
    os.environ["OPENAI_API_KEY"] = sk
    vectorstore = memory_store()
    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=1.0)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=1.0)
    Manager_agent = Manager_GPT.from_llm_and_tools(
        ai_name="MatterAI",
        ai_role="non",
        tools=tools_list,
        llm=llm,
        memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
        Expert_experience_path = Expert_experience_path,  # 设置专家经验数据库地址
        Expert_experience=Expert_experience,
        COB_in_the_loop=True
    )
    return Manager_agent


def CEO_to_Manager_parse(query: str) -> str:

    command_dict = ast.literal_eval(query)  # 把字符串转换为字典
    command_str = ""
    for i, v in command_dict.items():
        command_str += i + ": " + v + ","
    if "query" in command_str:
        command_str = command_str.replace("query", "")

    print("command_str:"+command_str)
    return command_str