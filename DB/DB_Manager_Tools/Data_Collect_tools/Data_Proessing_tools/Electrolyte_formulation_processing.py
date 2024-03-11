# 导入pandas和re模块
import pandas as pd
import re
import os
import sympy as sy
import numpy as np
import json
from DB.DB_Manager_Tools.Data_Collect_tools.Data_Proessing_tools.gogle_search_agent import search_agent
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
# Parse the lithium salt to get the content and name
def Li_salt_parse(Li_salts, components_dict):
    Li_salt_dict = {}
    Li_salt_split = Li_salts.split(",")
    for i in range(len(Li_salt_split)):
        Li_salt_number = "Li_salt_" + str(i + 1)
        if " M " in Li_salt_split[i]:
            aa = Li_salt_split[i].strip().split(" M ")
            mol_num = aa[0]
            Li_salt_name = aa[1].replace(".", "").replace(",", "")
        #             print("Li salt name:" + Li_salt_name)

        #         print(Li_salt_split[i])
        Li_salt_dict[Li_salt_number] = {"mol_num": float(Li_salt_split[i].strip().split(" ")[0]),
                                        "name": Li_salt_name}
        Li_salt_name = Li_salt_dict[Li_salt_number]["name"]

        #         print(components_dict[Li_salt_name])
        Li_salt_dict[Li_salt_number]["mass"] = float(components_dict[Li_salt_name]["molar mass"]) * \
                                               Li_salt_dict[Li_salt_number]["mol_num"]
        Li_salt_dict[Li_salt_number]["molar mass"] = components_dict[Li_salt_name]["molar mass"]
        Li_salt_dict[Li_salt_number]["density"] = components_dict[Li_salt_name]["density"]

        components_dict[Li_salt_name]["mol_num"] = float(Li_salt_split[i].strip().split(" ")[0])
        components_dict[Li_salt_name]["mass"] = Li_salt_dict[Li_salt_number]["mass"]
    return Li_salt_dict

#Parse the electrolyte to get the components and ratios
def electrolytes_parse(electrolytes):
    import re
    electrolyte_dict ={}
    pattern = r"\((.*?)\)"
    #得到比例
    result = re.findall(pattern,electrolytes)
#     print(result)
    ratio = result[-1].split(" ")[0].strip()
    ratio_type = result[-1].split(" ")[1].strip().split("/")[0].strip()
    #获取电解液组分名称
    elet_names = electrolytes.split(" (")[0].strip()
    elet_names_split = elet_names.split(":")
    single_ratio = ratio.split(":")
    for i in  range(len(elet_names_split)):
        singal_elet_name_num = "elet_name_" + str(i+1)
        electrolyte_dict[singal_elet_name_num] = {"name":elet_names_split[i].strip(),
                                             "ratio":float(single_ratio[i].strip()),
                                             "type":ratio_type}
    return electrolyte_dict


# 解析添加剂，得到组分与含量
def additives_parse(additives, components_dict):
    import re
    pattern = r"\d+(?:\.\d+)?"  # (?:...) 表示一个非捕获组，? 表示出现零次或一次
    content = re.findall(pattern, additives)  # 返回一个列表，包含所有匹配的数字
    additives_dict = {}
    additives_split = additives.split("and")
    for i in range(len(additives_split)):
        # 得到类型wt，vol，mol
        aa = additives_split[i].split(" ")
        print(aa)
        additives_con_type = [a for a in aa if "%" in a][0].split("%")[0]
        additives_number = "additives_" + str(i + 1)
        additives_name = additives_split[i].strip().split("%")[1].strip().replace(".", "")
        additives_dict[additives_number] = {"name": additives_name,
                                            "content": float(content[i]) / 100,
                                            "type": additives_con_type}

    return additives_dict


from sympy import *


def wt_calculate(electrolyte_dict, components_dict):
    # 首先判断有多少个组分
    components_num = len(electrolyte_dict.keys())
    # 假设混合体积为1L，设组分混合体积总和为1L
    # 生成一个整数序列
    r = range(components_num)
    # 转换成一个字符串
    unkonw_s = " ".join("_" + str(i + 1) for i in r)
    # 设未知数,创建符号变量,未知数为每个组分的体积
    symbol_str = ["_" + str(i + 1) for i in r]  # 变量的字符串形式
    [*var] = sy.symbols(unkonw_s)
    # 定义方程组
    str_eq0 = "+".join(str(i) for i in symbol_str)
    eq0 = sy.Eq(sympify(str_eq0), 1000)
    eq_dict = {}
    # 临时字典用于储存电解液名字与对应的未知数
    casual_name_dict = {}
    for i in range(components_num - 1):
        # 获取电解液编号1
        electrolyte_num_1 = "elet_name_" + str(i + 1)
        electrolyte_name_1 = electrolyte_dict[electrolyte_num_1].get("name")
        electrolyte_ratio_1 = electrolyte_dict[electrolyte_num_1].get("ratio")
        electrolyte_density_1 = components_dict[electrolyte_name_1].get("density")
        casual_name_dict[var[i]] = electrolyte_name_1

        # 获取电解液编号2
        electrolyte_num_2 = "elet_name_" + str(i + 2)
        electrolyte_name_2 = electrolyte_dict[electrolyte_num_2].get("name")
        electrolyte_ratio_2 = electrolyte_dict[electrolyte_num_2].get("ratio")
        electrolyte_density_2 = components_dict[electrolyte_name_2].get("density")
        casual_name_dict[var[i + 1]] = electrolyte_name_2
        #         electrolyte_molar_mass = components_dict[electrolyte_name].get("molar mass")
        eq_name = "eq" + str(i + 1)
        eq_dict[eq_name] = sy.Eq((var[i] * electrolyte_density_1) / (var[i + 1] * electrolyte_density_2),
                                 float(electrolyte_ratio_1) / float(electrolyte_ratio_2))
    eq_list = [v for k, v in eq_dict.items()]
    eq_list.append(eq0)
    #     print(eq_list)
    #     print(var)
    # 用 solve 函数求解
    volume = sy.solve(eq_list, var)
    # 得到的体积保存到components_dict中
    for k, v in casual_name_dict.items():
        components_dict[v]["volume"] = volume[k]
        components_dict[v]["mol_num"] = float(volume[k]) * float(components_dict[v]["density"]) / float(
            components_dict[v]["molar mass"])
        components_dict[v]["mass"] = float(components_dict[v]["molar mass"]) * float(components_dict[v]["mol_num"])
        for a, b in electrolyte_dict.items():
            if v == b["name"]:
                electrolyte_dict[a]["volume"] = volume[k]
                electrolyte_dict[a]["mol_num"] = components_dict[v]["mol_num"]
                electrolyte_dict[a]["mass"] = components_dict[v]["mass"]
    return components_dict


def vol_calculate(electrolyte_dict, components_dict):
    # 首先判断有多少个组分
    components_num = len(electrolyte_dict.keys())
    # 假设混合体积为1L，设组分混合体积总和为1L
    # 生成一个整数序列
    r = range(components_num)
    # 转换成一个字符串
    unkonw_s = " ".join("_" + str(i + 1) for i in r)
    # 设未知数,创建符号变量,未知数为每个组分的体积
    symbol_str = ["_" + str(i + 1) for i in r]  # 变量的字符串形式
    [*var] = sy.symbols(unkonw_s)
    # 定义方程组
    str_eq0 = "+".join(str(i) for i in symbol_str)
    eq0 = sy.Eq(sympify(str_eq0), 1000)
    eq_dict = {}
    # 临时字典用于储存电解液名字与对应的未知数
    casual_name_dict = {}
    for i in range(components_num - 1):
        # 获取电解液编号1
        electrolyte_num_1 = "elet_name_" + str(i + 1)
        electrolyte_name_1 = electrolyte_dict[electrolyte_num_1].get("name")
        electrolyte_ratio_1 = electrolyte_dict[electrolyte_num_1].get("ratio")
        electrolyte_density_1 = components_dict[electrolyte_name_1].get("density")
        casual_name_dict[var[i]] = electrolyte_name_1

        # 获取电解液编号2
        electrolyte_num_2 = "elet_name_" + str(i + 2)
        electrolyte_name_2 = electrolyte_dict[electrolyte_num_2].get("name")
        electrolyte_ratio_2 = electrolyte_dict[electrolyte_num_2].get("ratio")
        electrolyte_density_2 = components_dict[electrolyte_name_2].get("density")
        casual_name_dict[var[i + 1]] = electrolyte_name_2
        #         electrolyte_molar_mass = components_dict[electrolyte_name].get("molar mass")
        eq_name = "eq" + str(i + 1)
        eq_dict[eq_name] = sy.Eq(var[i] / var[i + 1], float(electrolyte_ratio_1) / float(electrolyte_ratio_2))
    eq_list = [v for k, v in eq_dict.items()]
    eq_list.append(eq0)
    #     print(eq_list)
    #     print(var)
    # 用 solve 函数求解
    volume = sy.solve(eq_list, var)
    # 得到的体积保存到components_dict中
    for k, v in casual_name_dict.items():
        components_dict[v]["volume"] = volume[k]
        components_dict[v]["mol_num"] = float(volume[k]) * float(components_dict[v]["density"]) / float(
            components_dict[v]["molar mass"])
        components_dict[v]["mass"] = float(components_dict[v]["molar mass"]) * float(components_dict[v]["mol_num"])
        for a, b in electrolyte_dict.items():
            if v == b["name"]:
                electrolyte_dict[a]["volume"] = volume[k]
                electrolyte_dict[a]["mol_num"] = components_dict[v]["mol_num"]
                electrolyte_dict[a]["mass"] = components_dict[v]["mass"]
    return components_dict


def mol_calculate(electrolyte_dict, components_dict):
    # 首先判断有多少个组分
    components_num = len(electrolyte_dict.keys())
    # 假设混合体积为1L，设组分混合体积总和为1L
    # 生成一个整数序列
    r = range(components_num)
    # 转换成一个字符串
    unkonw_s = " ".join("_" + str(i + 1) for i in r)
    # 设未知数,创建符号变量,未知数为每个组分的体积
    symbol_str = ["_" + str(i + 1) for i in r]  # 变量的字符串形式
    [*var] = sy.symbols(unkonw_s)
    # 定义方程组
    str_eq0 = "+".join(str(i) for i in symbol_str)
    eq0 = sy.Eq(sympify(str_eq0), 1000)
    eq_dict = {}
    # 临时字典用于储存电解液名字与对应的未知数
    casual_name_dict = {}
    for i in range(components_num - 1):
        # 获取电解液编号1
        electrolyte_num_1 = "elet_name_" + str(i + 1)
        electrolyte_name_1 = electrolyte_dict[electrolyte_num_1].get("name")
        electrolyte_ratio_1 = electrolyte_dict[electrolyte_num_1].get("ratio")
        electrolyte_density_1 = components_dict[electrolyte_name_1].get("density")
        electrolyte_molar_mass_1 = components_dict[electrolyte_name_1].get("molar mass")
        casual_name_dict[var[i]] = electrolyte_name_1

        # 获取电解液编号2
        electrolyte_num_2 = "elet_name_" + str(i + 2)
        electrolyte_name_2 = electrolyte_dict[electrolyte_num_2].get("name")
        electrolyte_ratio_2 = electrolyte_dict[electrolyte_num_2].get("ratio")
        electrolyte_density_2 = components_dict[electrolyte_name_2].get("density")
        casual_name_dict[var[i + 1]] = electrolyte_name_2
        electrolyte_molar_mass_2 = components_dict[electrolyte_name_2].get("molar mass")
        eq_name = "eq" + str(i + 1)
        eq_dict[eq_name] = sy.Eq((var[i] * electrolyte_density_1 / electrolyte_molar_mass_1) / (
                    var[i + 1] * electrolyte_density_2 / electrolyte_molar_mass_2),
                                 float(electrolyte_ratio_1) / float(electrolyte_ratio_2))
    eq_list = [v for k, v in eq_dict.items()]
    eq_list.append(eq0)
    #     print(eq_list)
    #     print(var)
    # 用 solve 函数求解
    volume = sy.solve(eq_list, var)
    # 得到的体积保存到components_dict中
    for k, v in casual_name_dict.items():
        components_dict[v]["volume"] = volume[k]
        components_dict[v]["mol_num"] = float(volume[k]) * float(components_dict[v]["density"]) / float(
            components_dict[v]["molar mass"])
        components_dict[v]["mass"] = float(components_dict[v]["molar mass"]) * float(components_dict[v]["mol_num"])
        for a, b in electrolyte_dict.items():
            if v == b["name"]:
                electrolyte_dict[a]["volume"] = volume[k]
                electrolyte_dict[a]["mol_num"] = components_dict[v]["mol_num"]
                electrolyte_dict[a]["mass"] = components_dict[v]["mass"]
    return components_dict

def obtain_elemental_ratios(components_dict):
    from molmass import Formula
    ele_com_dict = {}
    for component,attrib in components_dict.items():
        formula = components_dict[component]["chemical formula"]
        f = Formula(formula)
        composition = pd.DataFrame.from_dict(f.composition(), orient='index')
        for ele in composition.index:
            current_count = composition.loc[ele]["count"]*components_dict[component]["mol_num"]
            ele_symbol = composition.loc[ele]["symbol"]
            ele_com_dict[ele_symbol] = current_count + ele_com_dict.get(ele_symbol, 0)

    ele_count = np.array(list(ele_com_dict.values())).sum()
    # 使用map函数和lambda表达式来将每个值除以ele_count，得到一个迭代器
    values = map(lambda x: x / ele_count, ele_com_dict.values())
    # 使用dict函数和zip函数来创建一个新的字典，将原字典的键和新的值对应起来
    d2 = dict(zip(ele_com_dict.keys(), values))
    df = pd.DataFrame([d2])
    return df


def filling_component(recipes,components_dict):
    if " in " not in recipes:
        recipes = "0 M " + recipes.split(":")[0].strip() + " in " + recipes
    Li_salts = recipes.split("in")[0].strip()
    # 得到Li_salts的名称及其摩尔量
    Li_salt_dict = Li_salt_parse(Li_salts, components_dict)
    electrolytes = recipes.split(" in ")[1].strip()

    electrolytes = electrolytes.split("with")[0].strip()
    # 判断是否单个电解质
    if "(" in electrolytes:
        pass
    else:
        electrolytes = electrolytes + ":" + electrolytes + " (0:1 vol/vol)"

    # print("ss:" + electrolytes)
    # 解析电解质，得到组分与比列

    electrolyte_dict = electrolytes_parse(electrolytes)
    ratio_type = electrolyte_dict["elet_name_1"].get("type")
    if ratio_type == "wt":
        result = wt_calculate(electrolyte_dict, components_dict)
    elif ratio_type == "vol":
        result = vol_calculate(electrolyte_dict, components_dict)
    elif ratio_type == "mol":
        result = mol_calculate(electrolyte_dict, components_dict)

    if "with" in recipes:
        additives = recipes.split("with")[1].strip()
        additives_dict = additives_parse(additives, components_dict)
        for k, v in additives_dict.items():
            # 判断添加剂含量类型
            add_type = additives_dict[k].get("type")
            if add_type == "wt":
                # 获取电解质总质量
                mass_count = 0
                for num, inf in electrolyte_dict.items():
                    mass_count += inf["mass"]
                for num2, inf2 in Li_salt_dict.items():
                    mass_count += inf2["mass"]
                # 添加到添加剂,组分字典中
                add_mass = mass_count * additives_dict[k].get("content")
                additives_dict[k]["mass"] = add_mass
                add_name = additives_dict[k]["name"]
                components_dict[add_name]["mass"] = add_mass
                additives_dict[k]["mol_num"] = add_mass / float(components_dict[add_name]["molar mass"])
                components_dict[add_name]["mol_num"] = additives_dict[k]["mol_num"]
            elif add_type == "vol":
                # 获取电解质总体积
                vol_count = 0
                for num, inf in electrolyte_dict.items():
                    vol_count += inf["volume"]
                for num2, inf2 in Li_salt_dict.items():
                    vol_count += inf2["mol_num"] * inf2["molar mass"] / inf2["density"]
                # 添加到添加剂,组分字典中
                add_vol = vol_count * additives_dict[k].get("content")
                additives_dict[k]["volume"] = add_vol
                add_name = additives_dict[k]["name"]
                components_dict[add_name]["volume"] = add_vol
                additives_dict[k]["mol_num"] = add_vol * float(components_dict[add_name]["density"]) / float(
                    components_dict[add_name]["molar mass"])
                components_dict[add_name]["mol_num"] = additives_dict[k]["mol_num"]
    return components_dict


def ele_recipes_pro(path, column_name):
    origin_data = read_excel_data(path)
    new_data = pd.DataFrame()
    if column_name in origin_data.columns:
        for index, row in origin_data.iterrows():
            recipes = row[column_name]
            components_dict = search_agent(recipes)
            components_dict = json.loads(components_dict.replace("“", "\"").replace("”","\""))
            components_dict = filling_component(recipes,components_dict)
            conver_data = obtain_elemental_ratios(components_dict)
            #新建一个表格
            new_dataframe = pd.DataFrame(columns=origin_data.columns)
            new_dataframe = new_dataframe.append(row, ignore_index=True)
            del new_dataframe[column_name]
            # 合并数据集
            for columns_name in new_dataframe.columns:
                conver_data[columns_name] = new_dataframe[columns_name].values[0]
            new_data = pd.concat([new_data,conver_data],axis=0)
        new_data.fillna(0)
        save_path = os.getcwd() + "\\data_after_recipes_dig.csv"
        new_data.to_csv(save_path, index = False)
        return f"Digitization of the electrolyte formulas has been completed and the converted data is stored on {save_path}."
    else:
        return f"column '{column_name}' is not in DataFrame."

