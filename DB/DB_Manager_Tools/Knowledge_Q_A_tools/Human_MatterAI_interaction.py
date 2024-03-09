

from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Q_A_system import QA_Conversation


class ContinuousDialogue:
    def __init__(self, key_word):
        self.active = True  # 控制对话是否继续的标志
        self.key_word = key_word
    def start_conversation(self):
        print("Welcome to use the HMI to gain domain knowledge, you can enter your query below and when you need to finish, please enter exit.")
        while self.active:
            user_query = input("Your query: ")
            if user_query.lower() == 'exit':
                print("Conversation over.")
                self.active = False
            else:
                answer = QA_Conversation(user_query, self.key_word)
                print("MatterAI response:"+ answer.get("summary_answer"))

