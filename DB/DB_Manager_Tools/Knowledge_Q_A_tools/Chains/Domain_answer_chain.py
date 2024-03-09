from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from CEO.Base.CEO_sk import sk

prompt_template = """You are AI.\
Now you act a professor in the field of material science.\
You are great at answering questions in the field of materials science.\
Use the following pieces of context and conversation history to answer the question of Human at the end.\
Answers should be specific and not just emphasize the conclusion.If you don't know the answer,\
just say that you don't know, don't try to make up an answer.

Context:{context}.

The following is the conversation history between Human and You:
Conversation history:{history}.

The following is Human question:
Question: {question}.

Your answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "history", "question"]
)

llm_model = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, max_tokens=1000, openai_api_key=sk)
domain_chain = load_qa_chain(llm_model, chain_type="stuff", prompt=PROMPT, verbose=True, output_key="domain_answer")
