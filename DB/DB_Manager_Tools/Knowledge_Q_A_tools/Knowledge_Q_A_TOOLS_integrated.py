from DB.DB_Manager_Tools.Knowledge_Q_A_tools.General_Q_A_system import GeneralQAquery
from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Q_A_system_tool import DKquery
from DB.DB_Manager_Tools.Knowledge_Q_A_tools.Q_A_Communication_tool import DKCommunication

D_K_Communication_tool = DKCommunication()
General_Q_A_tool = GeneralQAquery()
D_K_tool = DKquery()
K_Q_A_tools_list = [D_K_tool, General_Q_A_tool, D_K_Communication_tool]

