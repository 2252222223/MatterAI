from CEO.Base.CEO_sk import sk
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter


def match_text(path,query):
    # 加载文件夹中的所有txt类型的文件
    loader = TextLoader(path,encoding= "utf-8")
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()
    # 初始化加载器
    text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = 1000,
        chunk_overlap  = 0,
        length_function = len,
    )
    split_docs = text_splitter.split_documents(documents)
    # 初始化 openai 的 embeddings 对象
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(split_docs, embeddings)
    docs = docsearch.similarity_search(query,k=10,include_metadata=True)
    return docs, split_docs