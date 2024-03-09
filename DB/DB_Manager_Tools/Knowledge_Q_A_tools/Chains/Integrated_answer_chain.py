from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from CEO.Base.CEO_sk import sk

llm = ChatOpenAI(model_name = "gpt-4-0613",temperature=0,max_tokens=1000, openai_api_key=sk)
template_string = """You are AI. You name is MatterAI. \
Now you act a professor in the field of material science.\
You are great at integrating  different viewpoints in the material field.\
Here are two answers that answer the question from different perspectives,\
and now you need to make an integration of the two answers.\

answer 1: ```{domain_answer}```

answer 2: ```{general_answer}```

question: ```{question}```


"""
three_prompt = ChatPromptTemplate.from_template(template_string)
# chain 4: input= summary, language and output= followup_message
Integrated_chain= LLMChain(llm=llm, prompt=three_prompt,
                      output_key="summary_answer"
                     )

