import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import SequentialChain
from CEO.Base.CEO_sk import sk
from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Chains.GPT4_answer_chain import GPT4_chain
from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Chains.Domain_answer_chain import domain_chain
from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Chains.Integrated_answer_chain import Integrated_chain

os.environ['OPENAI_API_KEY'] = sk
# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings(openai_api_key=sk)



from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10)


def simi_match(query,docsearch):
    # docs = docsearch.similarity_search(query, k=15, include_metadata=True)
    docs = docsearch.max_marginal_relevance_search(query, k=10, fetch_k=20, include_metadata=True)
    return docs


overall_chain = SequentialChain(
    chains=[domain_chain, GPT4_chain, Integrated_chain],
    input_variables=["input_documents", "history", "question"],
    output_variables=["domain_answer", "general_answer", "summary_answer"],
    verbose=True
)


def QA_Conversation(query, key_word, memory=memory):
    # 加载数据
    path = "E:\pycharm\\Matter AI-0705\\Knowledge_Acquisition_tools\\Domain Vector Library\\" + key_word +" vector_store"
    docsearch = Chroma(
        persist_directory=path,
        embedding_function=embeddings)
    history = memory.load_memory_variables({}).get("history")
    context = simi_match(query, docsearch)
    answer = overall_chain({"input_documents": context, "history": history, "question": query},
                           return_only_outputs=True)
    memory.save_context({"input": query},
                        {"output": answer.get("summary_answer")})
    return answer
