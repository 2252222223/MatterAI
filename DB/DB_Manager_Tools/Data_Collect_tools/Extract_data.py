from langchain.chat_models import ChatOpenAI
from CEO.Base.CEO_sk import sk
from DB.Base.Human_Review_Instruction import Human_Review
from kor.nodes import Object, Text, Number
from kor.extraction import create_extraction_chain
from langchain.prompts import PromptTemplate
import pandas as pd
import json


def printOutput(output):
    print(json.dumps(output,sort_keys=True, indent=3))


def creat_extract_chain(attributes):
    material_schema = Object(
        id="material",
        description="Information about a material",
        # Notice I put multiple fields to pull out different attributes
        attributes=attributes,
        many=False
    )
    five_template = """You are a data scientist who works with materials science datasets. \
    Your goal is to extract structured information from the user's input that matches the form described below. \
    When extracting information please make sure it matches the type information exactly. \
    Do not add any attributes that do not appear in the schema shown below.\
    NOTE,If this attributes is not found,YOU MUST OUTPUT None.\

    {type_description}

    {format_instructions}

    """

    five_template = five_template.format(type_description="{type_description}",
                                         format_instructions="{format_instructions}")
    instruction_template = PromptTemplate(
        input_variables=["type_description", "format_instructions"],
        template=five_template
    )
    #     llm = ChatOpenAI(model_name = "gpt-3.5-turbo",temperature=0,max_tokens=1000)
    llm = ChatOpenAI(openai_api_key=sk,
                     model_name="gpt-4-0613",
                     #     model_name="gpt-3.5-turbo",
                     temperature=0,
                     max_tokens=1000,
                     )
    #     chain = create_extraction_chain(llm, material_schema,instruction_template=instruction_template,verbose = True)
    chain = create_extraction_chain(llm, material_schema, instruction_template=instruction_template,
                                    encoder_or_encoder_class="csv")
    #     chain = create_extraction_chain(llm,material_schema,encoder_or_encoder_class="csv")
    return chain


def data_collect(extraction_chain,context):
    output = extraction_chain.run(text=(context))["data"]
    printOutput(output)
    return output

def check_df_columns(df):
    new_columns = []
    for columns in df.columns:
        if "." in columns:
            new_column_name = columns.split(".")
            new_columns.append(new_column_name[1])
        else:
            new_columns.append(columns)
    df.columns = new_columns
    return df


def collect_page_data(materials_feature,llm_response):  # 收集每一个段落的数据
    df = pd.DataFrame(llm_response["material"])
    df = check_df_columns(df)
    new_dataset = pd.DataFrame(columns=materials_feature.keys())  # 新建一个表格，用于储存数据
    if len(df.columns) == 1:
        sita = len(new_dataset)
        for index, row in df.iterrows():
            a = row.values[0].split(":")
            if a[0].lstrip() in new_dataset.columns:
                new_dataset.loc[sita, a[0].lstrip()] = a[1].replace(",", "")
    elif len(df.columns) == len(new_dataset.columns):
        new_dataset = pd.concat((new_dataset, pd.DataFrame(llm_response["material"])), axis=0)

    print("page_data:" + new_dataset)
    return new_dataset

def check_incorrect_listname(data):
    #因为可能存在material.的属性名，因此先整理表格
    raw_col = []
    new_col = []
    for i in data.columns:
        if "material." in i:
            new_col.append(i)
        else:
            raw_col.append(i)
    raw_data = data[raw_col]
    if len(new_col)>0:
        new_list_name =[x.split(".")[1] for x in new_col ]
        add_data = data[new_col]
        add_data.columns = new_list_name
        raw_data = pd.concat((raw_data,add_data),axis = 0)
    return raw_data

# 定义一个函数，检查一行中“None”或“-”的数量是否超过了总列数的一半
def check_row(row):
    # 获取总列数
    total_columns = len(row)
    # 获取“None”或“-”的数量
    none_or_dash = row.isin(["None", "-","NaN"]).sum()
    # 如果“None”或“-”的数量超过了总列数的一半，则返回True，否则返回False
    return none_or_dash > total_columns / 2



def check_data(data):
    data=check_incorrect_listname(data)
    #重置索引
    data.reset_index(drop=True, inplace=True)
    data = data.fillna("None")
    # 使用apply()方法将函数应用到dataframe的每一行，并得到一个布尔序列
    bool_index = data.apply(check_row, axis=1)

    # 使用drop()方法删除布尔序列为True的行，并返回一个新的dataframe
    new_df = data.drop(bool_index[bool_index].index)
    return new_df


def collect_paper_data(dataset,extraction_chain,simi_docs):
    for i in range(len(simi_docs)): #遍历一篇文章中可能的所有段落
        print(i)
        context = simi_docs[i].page_content
        llm_response = data_collect(extraction_chain,context)
        new_dataset = collect_page_data(llm_response)
        dataset = pd.concat((dataset,new_dataset),axis =0)
    dataset = check_data(dataset)#检查数据
    huamn_check = Human_Review()#人工审查
    note = f"Current result is: {dataset}. Enter yes if you want to keep all of them, or enter the number if you only want to keep a specific number of lines, noting that the numbering starts at 0. Such as 0,2"
    dataset = huamn_check.review_result(note,dataset)
    return dataset