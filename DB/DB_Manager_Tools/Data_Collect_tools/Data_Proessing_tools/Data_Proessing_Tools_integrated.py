from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.View_tabular_data_tool import View_Data
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.Chemical_formula_processing_tool import Chemical_formula_pro
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.Electrolyte_formulation_processing_tool import Electrolyte_composition_pro
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.Autonomous_feature_engineering_tool import Auto_Fe_Eng


View_Data_tool = View_Data()
Chemical_formula_pro_tool = Chemical_formula_pro()
Electrolyte_composition_pro_tool = Electrolyte_composition_pro()
Auto_Fe_Eng_tool = Auto_Fe_Eng()
Data_Proessing_tools_list = [View_Data_tool, Chemical_formula_pro_tool, Electrolyte_composition_pro_tool, Auto_Fe_Eng_tool]