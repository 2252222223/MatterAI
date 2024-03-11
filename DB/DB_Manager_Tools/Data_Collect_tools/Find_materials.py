from CEO.Base.CEO_sk import sk
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
engine = openai.Engine("davinci")
encoding = tiktoken.get_encoding("gpt2")


def read_text(path):
    with open(path, 'r',encoding="utf-8") as f: # 打开文件
        context = f.read() # 读取文件内容为字符串
    return context

def text_query(context, query):
    prompt_template = """
    Now you are a professor in the field of material science.\
    You are great at answering questions in the field of materials science based on papaer.\
    Use the following pieces of context to answer the question of Human at the end.\
    If you don't know the answer,just say that you don't know, don't try to make up an answer.

    Context:{context}.


    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The following is Human question:
    Question: {question}.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    customer_messages = prompt_template.format_messages(
                        question=query,
                        context=context)
    chat = ChatOpenAI(model_name = "gpt-3.5-turbo-0613",temperature=0,max_tokens=1000,openai_api_key= sk)
    customer_response = chat(customer_messages)
    return customer_response.content


def token_count(context,max_token):
    encoding = tiktoken.get_encoding("gpt2")
    text_token = len(encoding.encode(context))
    clip_text = context[:text_token if text_token<max_token else max_token]
    return clip_text


def gpt_query(query):
    prompt_template = """
    Now you are a professor in the field of material science.\
    You are great at answering human questions in the field of materials science.\

    The following is Human question:
    Question: {question}.

    """
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    customer_messages = prompt_template.format_messages(
                        question=query)
    chat = ChatOpenAI(model_name = "gpt-3.5-turbo-0613", temperature=0, max_tokens=1000,openai_api_key= sk)
    customer_response = chat(customer_messages)
    return customer_response.content


def find_materials(path):
    query = """What are the main materials discussed in the paper? If the material contains abbreviations, you must give the full name and the abbreviations.\
    You MUST need to FOCUS on the abbreviations of the material names first,and then match it correctly with the full name.\
    NOTE,you MUST answer only the full name and abbreviations of the material, abbreviations in parentheses after full name,and use a | as the delimiter.for example ```Li2MnO3(LMO)|Li1.8Co0.6V0.4O1.8F0.2(HLF20)```\
    If the abbreviation does not exist, you MUST answer only the full name of the material,Do not make up abbreviations， for example ```Li2MnO3|Li1.8Co0.6V0.4O1.8F0.2```"""

    context = read_text(path)
    context = token_count(context, 15000)
    customer_response = text_query(context, query)
    return customer_response

