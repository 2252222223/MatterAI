import os
from CEO.Base.CEO_sk import sk
os.environ["SERPAPI_API_KEY"] = "88504cebf38719c2d16a2ffbbb8a9b2424a65ef37b10676a94812b4975539c41"
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
import langchain


langchain.debug = False

def search_agent(recipes):
    tools = load_tools(["serpapi"])
    # 加载 OpenAI 模型
    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0, max_tokens=1000, openai_api_key=sk)
     # 加载 serpapi 工具
    tools = load_tools(["serpapi"])
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    template = f"""You are an AI who plays the role of a material scientist, working on advanced electrolyte design. Your task is to extract the composition from the electrolyte formula given by the user, and find the chemical formula, molar mass, and density of each component. To accomplish this task, you should:
    
    Split the formula into different components.
    Search online for the chemical formula, density, and molar mass of each component.
    NOTE, Your answer must be in json format without any additional information, ensuring that it can be parsed by json.load.
    Here is an example, marked with “””.
    “””user input: “1 M Lithium hexafluorophosphate in Ethylene carbonate:Diethyl carbonate (1:2 mol/mol) with 2 wt% Vinylene carbonate.” 
    Your answer: {{“Ethylene carbonate”:{{“density”:1.32,“molar mass”:88.062,“chemical formula”: “C3H4O3”}}, “Diethyl carbonate”:{{“density”:0.975,“molar mass”:118.13,“chemical formula”: “C5H10O3”}}, “Lithium hexafluorophosphate”:{{“density”:1.50,“molar mass”:151.91,“chemical formula”: “LiPF6”}}, “Vinylene carbonate”:{{“density”:1.360,“molar mass”:86.05,“chemical formula”: “C3H2O3”}},
    }}”””
    
    User input: {recipes}.
    Your answer:"""

    result = agent.run(template)

    return result
