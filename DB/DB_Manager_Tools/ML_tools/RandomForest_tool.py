from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from DB.DB_Manager_Tools.ML_tools.Base.Optimizer import optimizer_optuna
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.ML_tools.Base.Data_Preprocess import data_preprocessing

def RF_train(params, data, data2 = None, Predict=False):
    dataset = np.array(data)
    x = dataset[:, :-1]
    y = dataset[:, -1:].reshape(-1)
    if Predict is True:
        test_x = np.array(data2)
    n_splits = 5
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=2020)
    min_mae = []
    pre = 0
    for i, (trn_idx, val_idx) in enumerate(skf.split(x, y)):
        trn_x, trn_y = x[trn_idx], y[trn_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        rf_model = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
        rf_model.fit(trn_x, trn_y)
        if Predict is True:
            pre += rf_model.predict(test_x) / n_splits
        else:
            #                 min_r2.append(rf_model.score(val_x,val_y))
            val_mae = abs(abs(rf_model.predict(val_x)) - abs(val_y)).mean()
            min_mae.append(val_mae)
    if Predict is True:
        return pre
    else:
        return np.array(min_mae).mean()


def rf_objective(trial, data):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 10, 100, 5),
        'max_depth': trial.suggest_int("max_depth", 1, 10, 1),
    }

    loss = RF_train(params, data)

    return loss


def get_importance(params, data):
    dataset = np.array(data)
    x = dataset[:, :-1]
    y = dataset[:, -1:].reshape(-1)
    n_splits = 5
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=2020)
    for i, (trn_idx, val_idx) in enumerate(skf.split(x, y)):
        trn_x, trn_y = x[trn_idx], y[trn_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        rf_model = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
        rf_model.fit(trn_x, trn_y)
        break
    importances = rf_model.feature_importances_
    return importances


def get_prediction(params, data1, data2):
    pre_reslust = RF_train(params=params, data=data1, data2=data2, Predict=True)
    return pre_reslust


def RF_tool(task: str, file_path: str):
    # check task
    if task in ("Train_only", "Importance", "Prediction"):
        pass
    else:
        # 如果 task 参数不是期望的值，抛出异常或返回错误信息
        raise ValueError("The task must be Train_only, Importance or Prediction.")
    if task == "Train_only":
        new_path = data_preprocessing(file_path)
        data = read_excel_data(new_path)
        rf_best_params, rf_best_mae = optimizer_optuna(n_trials=100, algo="TPE", optuna_objective=rf_objective,
                                                       data=data)
        response = f"On dataset {file_path}, after Bayesian optimization, the best hyperparameter of the random forest model is {rf_best_params} and the best MAE metric is {rf_best_mae}."
        return response
    elif task == "Importance":
        new_path = data_preprocessing(file_path)
        data = read_excel_data(new_path)
        rf_best_params, rf_best_mae = optimizer_optuna(n_trials=50, algo="TPE", optuna_objective=rf_objective,
                                                       data=data)
        importances = get_importance(rf_best_params, data)
        importances_rank = np.argsort(-importances)

        response = f"The three most important features in dataset {file_path} are {data.columns[importances_rank[0]]},{data.columns[importances_rank[1]]},{data.columns[importances_rank[2]]} and their feature importance is {importances[importances_rank[0]]}, {importances[importances_rank[1]]} and {importances[importances_rank[2]]} respectively."
        return response
    elif task == "Prediction":
        train_dateset, prediction_dateset = file_path.split(",")
        train_dateset, prediction_dateset = data_preprocessing(train_dateset, prediction_dateset)
        data1 = read_excel_data(train_dateset)
        data2 = read_excel_data(prediction_dateset.lstrip())
        rf_best_params, rf_best_mae = optimizer_optuna(n_trials=50, algo="TPE", optuna_objective=rf_objective,
                                                       data=data1)
        pre_reslust = get_prediction(rf_best_params, data1, data2)
        response = f"The prediction of random forest is {pre_reslust}."
        return  response


class R_F_PreSchema(BaseModel):

    task: str = Field(description="Should be one of the three, Train_only, Importance, Prediction. Train_only represents that the model is only trained and returns the best hyperparameters for the random forest model and the corresponding evaluation metrics. Importance represents that the model returns only the importance of the features. Prediction represents that the model gives predictions for the test set.")
    f_path: str = Field(description = "Should be a dataset address. When the task is Train_only and Importance, only the file address of a training set needs to be entered. When the task is Prediction, the inputs are Training Set and Test Set, separated by `,`.")


class RF_Regression(BaseTool):
    name = "Random_Forest_Regression"
    description = "Very useful when you want to use random forest regression algorithms."
    args_schema: Type[BaseModel] = R_F_PreSchema

    def _run(self, task: str, f_path: str) -> str:

        return RF_tool(task, f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")