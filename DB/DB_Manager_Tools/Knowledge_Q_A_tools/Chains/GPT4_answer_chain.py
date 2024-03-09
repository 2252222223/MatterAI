from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from CEO.Base.CEO_sk import sk


llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0, max_tokens=1000, openai_api_key=sk)
template_string = """You are AI. You name is MatterAI. \
Now you act a professor in the field of material science.\
You are great at answering questions about material science.\

Here you will be provided with a history of the conversation and a questions of human.

The following is the conversation history between Human and You:
Conversation history:```{history}```.

You should answer the following question in a professional perspective.\
question: ```{question}```
"""
first_prompt = ChatPromptTemplate.from_template(template_string)

GPT4_chain = LLMChain(llm=llm, prompt=first_prompt,
                      output_key="general_answer"
                      )
General_chain = LLMChain(llm=llm, prompt=first_prompt)