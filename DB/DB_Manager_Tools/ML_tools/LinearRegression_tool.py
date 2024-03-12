import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from sklearn.metrics import r2_score,mean_absolute_error
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
from DB.DB_Manager_Tools.ML_tools.Base.Data_Preprocess import data_preprocessing

class L_R_PreSchema(BaseModel):

    f_path: str = Field(description="Should be a tabular address ending in csv or xlx.")
    metrics: str = Field(description="Optional, Should be a model evaluation metric, either MAE or R2, default R2.")


def Line_Reg(f_path: str, metrics: Optional[str] = "MAE", seed=42) -> str:
    data_path = data_preprocessing(f_path)
    data = read_excel_data(data_path)
    datasets = np.array(data)
    n_splits = 4
    skf = KFold(n_splits=n_splits, shuffle=True)
    Metr =[] #评估指标
    x = datasets[:, :-1]
    y = datasets[:, -1:]
    cor = np.zero((1,x.shape[0]))
    intercept = 0
    for i, (trn_idx, val_idx) in enumerate(skf.split(x, y)):
        trn_x, trn_y = x[trn_idx], y[trn_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        # 创建并拟合线性回归模型
        model = LinearRegression()
        model.fit(trn_x, trn_y)
        if metrics == "R2":
            r_squared = r2_score(model.predict(val_x), val_y)
            Metr.append(r_squared)
        elif metrics == "MAE":
            r_mae = mean_absolute_error(model.predict(val_x), val_y)
            Metr.append(r_mae)
        cor += model.coef_/n_splits
        intercept += model.intercept_/n_splits

    Metr_mean = np.array(Metr).mean()
    #表达式
    y = ""
    for i in range(len(cor[0])):
        if cor[0][i] >0:
            y += str(cor[0][i]) + "*" + data.columns[i] + "+"
        else:
            y += str(cor[0][i]) + "*" + data.columns[i]
    y = y + str(intercept/n_splits) if y.endswith("+") else "+" + str(intercept)
    response = f"""The linear regression expression for dataset {f_path} is {y}, and its {metrics} metric is {Metr_mean}."""

    return response


class Line_Regression(BaseTool):
    name = "Line_Regression"
    description = "Very useful when you want to use linear regression algorithms."
    args_schema: Type[BaseModel] = L_R_PreSchema

    def _run(self, f_path: str, metrics: Optional[str]) -> str:

        return Line_Reg(f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

