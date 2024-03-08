from typing import List, Optional
from pydantic import ValidationError
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain_experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain_experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.memory import ChatMessageHistory
from langchain.schema import (
    BaseChatMessageHistory,
    Document,
)
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever
from DB.Base.Manager_prompt import Manager_Prompt
from CEO.Base.COB_intervention import COBInputRun
from CEO.Base.Expert_experience import expert_experience_match
from Expert_experience_digitization_module.Experience_storage import expert_guidance_exp_storage, practice_exp_storage

class Manager_GPT:
    """Agent class for interacting with Manager-GPT."""
    def __init__(
            self,
            ai_name: str,
            memory: VectorStoreRetriever,
            chain: LLMChain,
            output_parser: BaseAutoGPTOutputParser,
            tools: List[BaseTool],
            Expert_experience_path: str,
            Expert_experience: bool,
            feedback_tool: Optional[COBInputRun] = None,
            chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool
        self.chat_history_memory = chat_history_memory or ChatMessageHistory()
        self.Expert_experience_path = Expert_experience_path
        self.Expert_experience = Expert_experience
    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        COB_in_the_loop: bool = False,
        Expert_experience_path: str = None,
        Expert_experience: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ):
        prompt = Manager_Prompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "cob_command", "expert_experience", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        human_feedback_tool = COBInputRun() if COB_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            Expert_experience_path,
            Expert_experience,
            feedback_tool=human_feedback_tool,
            chat_history_memory=chat_history_memory,
        )

    def run(self, goals: List[str]) -> str:
        def execute_tool():
            self.chat_history_memory.add_message(HumanMessage(content=user_input))
            self.chat_history_memory.add_message(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                return action
            if action.name in tools:
                tool = tools[action.name]
                try:
                    # print(action.args)
                    observation = tool.run(action.args)
                    # print(observation)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            # if self.feedback_tool is not None:
            #     feedback = f"\n{self.feedback_tool.run('Input: ')}"
            #     if feedback in {"q", "stop"}:
            #         print("EXITING")
            #         return "EXITING"
            #     memory_to_add += feedback

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.chat_history_memory.add_message(SystemMessage(content=result))

            return action
        # user_input = (
        #     "Determine which next command to use, "
        #     "and respond using the format specified above:"
        # )
        # user_input = ("First, it determines whether the task has been completed, and if it has, it ends immediately. If it is not completed, Determine which next command to use, and respond using the format specified above:")
        user_input = ("Determine which next command to use, and respond using the format specified above.")
        cob_command =("")
        if self.Expert_experience is True:

            expert_experience_ma = expert_experience_match(goals, self.Expert_experience_path)
            expert_experience_input = f"The following experiences from similar tasks will help you immensely, so please focus on them. Related Experience:{expert_experience_ma}"
            expert_experience = (expert_experience_input)
        else:
            expert_experience =("")
        # Interaction Loop
        loop_count = 0
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.chat_history_memory.messages,
                memory=self.memory,
                user_input=user_input,
                cob_command =cob_command,
                expert_experience=expert_experience,
            )
            # Print Assistant thoughts
            # print(assistant_reply)
            if self.feedback_tool is not None:
                DB_decision = assistant_reply
                report_to_COB = f"Honorable COB, the following is the Departmental Manager decision report:" \
                                f"{DB_decision}" \
                                f"" \
                                f"Now,please make your instructions:"
                feedback = f"\n{self.feedback_tool.run(report_to_COB)}"


                if "YES" in feedback.upper():
                    stroge_path = self.Expert_experience_path.rsplit('\\', 1)[0] +"\experience.txt"
                    if len(self.chat_history_memory.messages)>0:
                        last_response = self.chat_history_memory.messages[-1].content
                    else:
                        last_response = ""

                    print("上一次工具响应：" + last_response)
                    cob_command = ("")

                    try:
                        expetr_guidance
                    except NameError:
                        expetr_guidance = False

                    if expetr_guidance:
                        expert_guidance_exp_storage(goals[0], last_response, expetr_guidance_content, DB_decision, stroge_path)
                        expetr_guidance = False
                    else:
                        practice_exp_storage(goals[0],last_response, DB_decision,stroge_path)
                    action =execute_tool()
                    if action.name == FINISH_NAME:
                        return action.args["response"]
                else:
                    cob_instructions = f"The following are the CEO's instructions, which you must obey unconditionally. Instructions:{feedback}"
                    expetr_guidance = True
                    expetr_guidance_content = feedback
                    cob_command = (cob_instructions)
            else:
                action =execute_tool()
                if action.name == FINISH_NAME:
                    return action.args["response"]