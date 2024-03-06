from langchain.chat_models import ChatOpenAI
import os
from CEO.Base.CEO_sk import sk, search_key
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from CEO.Base.CEO_Auto_gpt import CEO_GPT
import faiss
from langchain.vectorstores import FAISS
from CEO.CEO_Manager_Tools.CEO_Manger_Tool_integrated import CEO_tools_list

os.environ["SERPAPI_API_KEY"] = search_key
os.environ["OPENAI_API_KEY"] = sk

def memory_store():
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    return vectorstore


tools = CEO_tools_list

vectorstore = memory_store()
llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)

Expert_experience_path = "E:\pycharm\\MatterAI-0816-only-test\\CEO\\CEO_Manager_Tools\\Expert_experience\\Expert_experience vector_store"

CEO_agent = CEO_GPT.from_llm_and_tools(
    ai_name="the CEO",
    ai_role="you need to organize your subordinates to collaborate in accomplishing goal.",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
    Expert_experience_path = Expert_experience_path,  # 设置专家经验数据库地址
    Expert_experience =False,
    COB_in_the_loop=True,  # Set to True if you want to add feedback at each step.
)

import langchain

# print(CEO_agent.Expert_experience)
langchain.debug = True

query = """"""

CEO_agent.run([query])
