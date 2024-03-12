import numpy as np
from DB.DB_Manager_Tools.ML_tools.Base.Optimizer import optimizer_optuna
from DB.DB_Manager_Tools.ML_tools.XGB_tool import xgb_objective
import xgboost as xgb
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
import shap
from DB.DB_Manager_Tools.ML_tools.Base.Data_Preprocess import data_preprocessing

def single_explain(background_data, single_data):
    xgb_best_params, _ = optimizer_optuna(n_trials=100, algo="TPE", optuna_objective=xgb_objective,
                                                     data=background_data)
    X, y = background_data.iloc[:, :-1], background_data.iloc[:, -1:]
    X_test = single_data.iloc[:1, :]
    dataTrain = xgb.DMatrix(X, y)
    dataTest = xgb.DMatrix(X_test)
    # train XGBoost model
    model = xgb.train(params=xgb_best_params, dtrain=dataTrain, num_boost_round=50000, callbacks=[xgb.callback.EarlyStopping(50)])
    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dataTest)
    # 画图
    # visualize the first prediction's explaination with default colors
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])
    #预测值
    prediction = explainer.expected_value
    prediction += np.sum(shap_values[0, :])

    #积极影响特征
    postive_index = np.where(shap_values[0, :] > 0)
    postive_feature = ", ".join(X_test.iloc[0, :].index[postive_index])
    postive_feature_shap = ", ".join([str(round(x, 2)) for x in shap_values[0, :][postive_index]])

    #消极影响
    negative_index = np.where(shap_values[0, :] <= 0)
    negative_feature = ", ".join(X_test.iloc[0, :].index[negative_index])
    negative_feature_shap = ", ".join([str(round(x, 2)) for x in shap_values[0, :][negative_index]])
    response = f"Compared to the base value of {explainer.expected_value:.2f}, the prediction of the sample is {prediction:.2f}, which is due to the joint influence of the features. The contribution of each feature was quantified using SHAP values. Features {postive_feature} have positive impact on the results with Shap values of {postive_feature_shap} while features {negative_feature} have negative impact on the results with Shap values of {negative_feature_shap}."
    return response


def feature_importance(background_data):
    xgb_best_params, _ = optimizer_optuna(n_trials=100, algo="TPE", optuna_objective=xgb_objective,
                                                     data=background_data)
    X, y = background_data.iloc[:, :-1], background_data.iloc[:, -1:]
    dataTrain = xgb.DMatrix(X, y)
    # train XGBoost model
    model = xgb.train(params=xgb_best_params, dtrain=dataTrain, num_boost_round=5000)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    importances = np.abs(shap_values).mean(0)
    index = np.argsort(-importances)[:3]
    most_feature = ", ".join(X.columns[index])
    most_feature_shap = ", ".join([str(round(x, 2)) for x in importances[index]])
    response = f"The three features that have the most important impact on the results are{most_feature} and their importance is {most_feature_shap}."
    return response


def SHAP_tool(task: str, train_dateset: str, prediction_dateset: Optional[str] = None):

    # check task
    if task in ("Sample_explanation", "Feature_Importance"):
        pass
    else:
        # 如果 task 参数不是期望的值，抛出异常或返回错误信息
        raise ValueError("The task must be Sample_explanation or Feature_Importance.")

    if task =="Sample_explanation":
        train_dateset, prediction_dateset = data_preprocessing(train_dateset,prediction_dateset)
        data1 = read_excel_data(train_dateset)
        data2 = read_excel_data(prediction_dateset)
        response =single_explain(data1, data2)
        return response
    elif task == "Feature_Importance":
        train_dateset = data_preprocessing(train_dateset)
        data = read_excel_data(train_dateset)
        importance = feature_importance(data)
        return importance


class Shap_PreSchema(BaseModel):

    task: str = Field(description="Should be one of the two, Sample_explanation or Feature_Importance. Sample_explanation represents the explanation of the effect of each feature in the sample on the predicted values of the model using the SHAP values. Feature_Importance represents the calculation of the average importance of each feature.")
    train_dateset: str = Field(description="Should be a dataset address.When the task is Feature_Importance, only the file address of a training set needs to be entered. ")
    prediction_dateset: Optional[str] = Field(description="Should be a dataset address. When the task is Sample_explanation,should input a table file to be interpreted.")


class SHAP(BaseTool):
    name = "SHAP"
    description = "Useful when you need to utilize explainable machine learning algorithms."
    args_schema: Type[BaseModel] = Shap_PreSchema

    def _run(self, task: str, train_dateset: str, prediction_dateset: Optional[str] = None) -> str:

        return SHAP_tool(task, train_dateset, prediction_dateset)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")
