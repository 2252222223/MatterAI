from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from kor.nodes import Object, Text, Number
from CEO.Base.CEO_sk import sk
import json
from DB.Base.Human_Review_Instruction import Human_Review

def parse_llm_output(four_customer_response):

    # kk = four_customer_response.replace("\"","").replace("}","").replace("{","").replace("\n    ","").replace("\n","").replace("'","")
    # materials_feature = {}
    # print(kk)
    # for c in kk.split(","):
    #     materials_feature[c.split(":")[0].replace("'","").replace(" ","")] =c.split(":")[1].replace("'","").lstrip()
    # return materials_feature
    four_customer_response = four_customer_response.replace("'", '"')
    materials_feature = json.loads(four_customer_response)
    return materials_feature

def attributes_make(materials_feature):
    attributes = []
    for k,v in materials_feature.items():
        if "Number" in v:
            kk = Number(id= k,description=v.split(".")[0],many  = False)
        if "String" in v:
            kk = Text(id= k,description=v.split(".")[0],many  = False)
        attributes.append(kk)
    return attributes


def generate_attributes(second_llm_output):
    four_chat= ChatOpenAI(model_name = "gpt-4-0613",temperature=0,max_tokens=1000,verbose=True, openai_api_key= sk)
    four_template = """
    You are now a professor in the field of Li-ion batery.\
    Your goal is to extract the terminology from the user's input and explain it, \
    as concisely and professionally as possible.\
    These terminology will be used to make datasets for machine learning training, \
    so you need to judge whether the attribute corresponding to the terminology is a string or a number, and must USE ** to mark it.\
    If the property is a range, split it into an upper and lower limit, e.g. Voltage Range should be split into low_voltage and high_voltage.\
    The information of your output makes sure it can be parsed by json.loads. All keys and values must be enclosed in double quotes. All keys must be lowercase.\
    Do NOT add any additional explanations.


    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Here,give you a answer example:
    % START EXAMPLES
    {example}
    % END EXAMPLES
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The following is user input:
    % USER INPUT:
    {user_input}

    YOUR RESPONSE:
    """
    four_example = {"material_type": "The name of the material.**String",
    "low_voltage": "The low end of a working voltage range.**Number",
    "high_voltage": "The high end of a working voltage range.**Number",
    "discharge_capacity": "The initial discharge capacity of a working voltage range.**Number",
    "discharge_rate": "The discharge rate of the battery.**Number"}
    four_prompt_template = ChatPromptTemplate.from_template(four_template)
    four_customer_messages = four_prompt_template.format_messages(
                        example=four_example,
                        user_input=second_llm_output)
    four_customer_response = four_chat(four_customer_messages)
    reviewer = Human_Review()
    note = f"Current result is: {four_customer_response.content}, enter yes if you agree, or enter a new result if you don't agree."
    llm_response = reviewer.review_result(note, four_customer_response.content)
    materials_feature = parse_llm_output(llm_response)
    attributes = attributes_make(materials_feature)
    return attributes, materials_feature
