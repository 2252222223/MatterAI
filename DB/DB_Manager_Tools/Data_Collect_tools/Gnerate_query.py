from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from CEO.Base.CEO_sk import sk


def generate_query(materials_name, query):
    second_template = """You are a professor in the field of material science.\
    Your goal is to integrate the two texts into a more explicit sentence,\
    where the first text is the name of the material, where the different names are split by |,\
    and the second text is the user's input,\
    containing the material properties that the user cares about, but the name of the material is not included.\
    Your output makes it clear what material the user cares about, and its corresponding properties.

    the firest text:{materials_name}

    the second text:{query}

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Here,give you a answer example:```

    the firest text: Li2MnO3
    the second text: I am concerned about the initial discharge capacity of the material, and the corresponding discharge voltage range.

    Your answer: The initial discharge capacity of Li2MnO3, and the corresponding discharge voltage range.```
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Your answer:
    """
    second_prompt = ChatPromptTemplate.from_template(second_template)
    second_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0, openai_api_key= sk)
    second_customer_messages = second_prompt.format_messages(materials_name=materials_name, query = query)
    second_llm_output = second_llm(second_customer_messages)

    return second_llm_output.content