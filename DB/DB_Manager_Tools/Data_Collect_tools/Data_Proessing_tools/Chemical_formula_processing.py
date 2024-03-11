import re
import pandas as pd
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data


def split_chemical_formula(f_path, column):

    df = read_excel_data(f_path)
    # 新建一个DataFrame来存储结果
    result_df = pd.DataFrame()
    # 遍历原始DataFrame中的每一行
    for index, row in df.iterrows():
        # 提取化学式
        formula = row[column]
        # 分解化学式为元素和数量
        elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        # 创建一个临时字典来存储元素和数量
        temp_dict = {}
        for element, count in elements:
            # 如果没有指定数量，默认为1
            count = int(count) if count else 1
            temp_dict[element] = count
        # 将临时字典转换为DataFrame并添加到结果DataFrame中
        temp_df = pd.DataFrame(temp_dict, index=[index])
        result_df = result_df.append(temp_df)

    # 用0填充NaN值
    result_df = result_df.fillna(0).astype(int)
    # 将原始DataFrame的其他列合并到结果DataFrame中
    result_df = pd.concat([result_df,df.drop(columns=[column])], axis=1)
    # 返回结果DataFrame
    save_path = f_path.split(".")[0] + "formula_convert.xls"
    result_df.to_csv(save_path,index=False)
    return f"Digital conversion of the chemical formula for {column} has been completed and the converted file is saved on the {save_path}."
