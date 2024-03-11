from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from DB.Base.Human_Review_Instruction import Human_Review


def find_missvalue_and_generate_query(text :str):
    template = """
    You are a material scientist. \
    You have a dictionary of electrolyte properties, but some values are missing and marked as None. \
    You need to generate a query command that uses the some existing values to search for the missing ones in a text document.\ 
    You need to ignore the "_" in the dictionary keys. \
    Note that the generated query can only contain one missing attribute at a time.\
    If there are no missing values, you can only answer "No missing".

    ```Some example:
    {example}

    ```    
    User input:{text}
    Your answer:
    """
    huaman_check = Human_Review()
    example = """User input: "{'electrolyte_name': 'NMD', 'electrolyte_composition': 'None', 'current_density': '5 mAh -1', 'coulombic_efficiency': '98.3'}"                                
    Your answer: electrolyte_composition: What is the composition of NMD electrolyte?"""
    note = f"The current example is:{example}.enter yes if you agree, or enter a new example if you don't agree."
    example = huaman_check.review_result(note,example)
    second_prompt = ChatPromptTemplate.from_template(template)
    second_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    second_customer_messages = second_prompt.format_messages(example=example,text = text)
    second_llm_output = second_llm(second_customer_messages)
    return second_llm_output.content

#截断paper长度，避免超出token上限
def paper_clip(paper:str):
    import tiktoken
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    paper_token = len(encoding.encode(paper))
    max_token_num=7500
    prompt_token = 300
    clip_paper_index = int(len(paper)*(max_token_num-prompt_token)/paper_token)
    clip_paper = paper[:clip_paper_index]
    return clip_paper


# 根据问题在全文中寻找答案
def answer_query_base_text(new_query, paper, query_attribute,response_schema):
    template = """
    You are a material scientist.\
    You will receive two inputs: a user query and an academic paper. \
    Your task is to answer the user query based on the information in the academic paper. \
    Your answer must refer to the following format: The {query_attribute} is {response_schema}.\
    Note that if you cannot find the answer in the paper, you must answer "I don't know" without any explanation.\
    Do not make up any information that is not in the paper.\

    User query:{new_query}.
    Academic paper:{paper}.
    Paper end.


    Your answer:
    """
    paper = paper_clip(paper)
    second_prompt = ChatPromptTemplate.from_template(template)
    second_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    second_customer_messages = second_prompt.format_messages(new_query=new_query, paper=paper,query_attribute = query_attribute,
                                                             response_schema=response_schema)
    second_llm_output = second_llm(second_customer_messages)
    return second_llm_output.content

def custom_query_base_text(new_query, paper):
    template = """
    You are a material scientist.\
    You will receive two inputs: a user query and an academic paper. \
    Your task is to answer the user query based on the information in the academic paper. \
    Note that if you cannot find the answer in the paper, you must answer "I don't know" without any explanation.\
    Do not make up any information that is not in the paper.\

    User query:{new_query}.
    Academic paper:{paper}.
    Paper end.


    Your answer:
    """
    paper = paper_clip(paper)
    second_prompt = ChatPromptTemplate.from_template(template)
    second_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    second_customer_messages = second_prompt.format_messages(new_query=new_query, paper=paper)
    second_llm_output = second_llm(second_customer_messages)
    return second_llm_output.content



# 判断是否self_query是否找到了答案
def answer_parse(new_query, llm_response):
    template = """
    In this task, you will be given a user's query and an answer. 
    Your responsibility is to judge whether the answer has already solved the user's query. 
    You only need to answer yes or no. 
    Note that if the answer indicates that it does not appear, your response can only be no.
    You do not need to explain your answer.

    user's query:{new_query}.
    answerr:{answer}.

    Your answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    customer_messages = prompt.format_messages(new_query=new_query, answer=llm_response)
    llm_output = llm(customer_messages)
    return llm_output.content

def get_full_paper(split_docs):
    paper = ""
    for i in range(len(split_docs)):
        paper += split_docs[i].page_content + "\n "
    return paper

def fill_data_None(data,paper):
    data_jason = data.to_json(orient="records", lines=True).split("\n")#将当前收集的数据转换为jason格式,如果是多行则转换为一个列表
    human_check = Human_Review()
    #重置索引
    data.reset_index(drop=True, inplace=True)
    for i in range(len(data_jason)):
        if "None" in data_jason[i]:
            #根据缺失值生成问题
            self_query = find_missvalue_and_generate_query(data_jason[i])
            #拆分问题，得到对应的待查询属性及对应的查询问题
            print("self_query:" + self_query)
            note = f"The current self query is:{self_query}.enter yes if you agree, or enter a new query if you don't agree."
            self_query = human_check.review_result(note,self_query)
            if ":" in self_query:
                query_attribute, next_query = self_query.split(":")
                #根据查询在全文寻找答案
                response_schema = "0.1 m LiDFP in ethylene carbonate (EC) and dimethyl carbonate (DMC) (5:5 by weight ratio)"
                note = f"The currently expected response format is:{response_schema}.enter yes if you agree, or enter a new response_schema if you don't agree."
                response_schema = human_check.review_result(note, response_schema)

                sentence = answer_query_base_text(next_query,paper,query_attribute, response_schema)
                print("next_query:" + next_query)
                print("sentence:" + sentence)
                #引入一个监督者角色，判断答案是否符合要求
                # answer_parse_result = answer_parse(self_query,sentence)
                note = f"The current self query is:{self_query}. if you think it's correct, please enter correct, if you don't agree, please enter no."
                answer_parse_result = human_check.review_result(note, sentence)
                #如果答案符合要求。在表格中填充
                if answer_parse_result.upper() == "YES":
                    parts = sentence.split("is",1)
                    find_attribute =parts[1].strip()
                    data.loc[i,query_attribute] = find_attribute
    return data


def fill_abbreviations(data,paper):
    for col in data.columns:
        for i, val in enumerate(data[col]):
            # 打印当前值
            print(f'Current column {col},row {i} is {val}')
            # 询问用户是否需要更改此值
            user_input = input('Is this value correct？(y/n): ')
            if user_input.lower() == 'n':
                # 如果用户输入'n'，请求新的值
                new_query = input('Input your query: ')
                sentence = custom_query_base_text(new_query, paper)

                new_val = input(f'This is the result :{sentence} ,which obtained by residing the custom query, now please enter the full content.: ')
                # 更新DataFrame
                data.at[i, col] = new_val
    return data