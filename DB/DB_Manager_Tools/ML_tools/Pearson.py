#!/usr/bin/env python
# coding: utf-8



from langchain.tools import BaseTool
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data


def observation(a:pd.core.frame.DataFrame)-> str:
    answer = ""
    # for i in a.columns:
    #     other_features = ""
    #     values = ""
    #     res = "The correlation coefficients between {featureA} and {other_features} is {values}respectively. "
    #     for k in a.columns:
    #         if i != k:
    #             other_features += "{feature}, ".format(feature=k)
    #             values += "{:.2}, ".format(a.loc[i, k])
    #     answer += res.format(featureA=i, other_features=other_features, values=values)
    final = "The three features that have the most influence on feature {curious_feature} are {most_influence} respectively. "
    kk = a.sort_values(by=a.columns[-1], key=abs, ascending=False).index[1:4 if len(a.columns) > 4 else -1]
    most_influence = ", ".join(kk)
    end = final.format(curious_feature=a.columns[-1], most_influence=most_influence)
    answer += "" + end

    return answer



class HotMaPDraw(BaseTool):
    name = "hot_map_draw"
    description = "Useful when you need to know the Pearson correlation coefficient.Input:Address of a excel file"

    def _run(self, query: str) -> str:
        data = read_excel_data(query)
        r_pearson = data.corr()
        plt.figure(dpi=120)
        sns.heatmap(data=r_pearson,#矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签
                cmap=plt.get_cmap('jet_r'),
                annot=True,
                vmin=-0.8,#图例（右侧颜色条color bar）中最小显示值 
                vmax=0.8,#图例（右侧颜色条color bar）中最大显示值
                annot_kws={'size':8,'weight':'normal', 'color':'black'},#设置格子中数据的大小、粗细、颜色
                linewidths=1,#每个格子边框宽度，默认为0
                linecolor='white',#每个格子边框颜色,默认为白色
                cbar=True,
                fmt=".2f")
        # plt.title('所有参数默认')
        plt.savefig('./hot_map.png', dpi=600, format='png',bbox_inches='tight')
        print("\nHOT_MAP_TOOL query: " + query)
        answer = observation(r_pearson)
        answer_start = "Correlation analysis has been completed, Pearson correlation coefficient plots have been plotted and saved in . /hot_map.png."
        final_answer = answer_start +" From the figure," + answer + " The analysis of correlations between features has been completed."

        return final_answer
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

