
from DB.DB_Manager_Tools.Data_Collect_tools.Data_collect_tool import Data_Collect
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.Data_Proessing_agent import Custom_Dtat_Proess_Tool

D_C_tool = Data_Collect()
Data_proess_tool = Custom_Dtat_Proess_Tool()
Data_Collect_tools_list = [D_C_tool, Data_proess_tool]