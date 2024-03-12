from sklearn.model_selection import KFold
import numpy as np
from DB.DB_Manager_Tools.ML_tools.Base.Optimizer import optimizer_optuna
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
import xgboost as xgb
from DB.DB_Manager_Tools.ML_tools.Base.Data_Preprocess import data_preprocessing

def xgb_train(params, data, data2=None, Predict=False):
    dataset = np.array(data)
    x = dataset[:, :-1]
    y = dataset[:, -1:]
    if Predict is True:
        test_x = np.array(data2)
    n_splits = 5
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=2020)
    min_mae = []
    pre = 0

    evals_result = {}
    for i, (trn_idx, val_idx) in enumerate(skf.split(x, y)):
        trn_x, trn_y = x[trn_idx], y[trn_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        dataTrain = xgb.DMatrix(trn_x, trn_y)
        dataVal = xgb.DMatrix(val_x, val_y)
        watchlist = [(dataVal, 'test'), (dataTrain, 'train')]

        if Predict is True:
            dataTest = xgb.DMatrix(test_x)
            bst = xgb.train(params=params, dtrain=dataTrain, num_boost_round=50000, evals=watchlist,
                            callbacks=[xgb.callback.EarlyStopping(50)])
            pre += bst.predict(dataTest) / n_splits
            return pre

        else:
            bst = xgb.train(params=params, dtrain=dataTrain, num_boost_round=3000, evals=watchlist,
                            evals_result=evals_result, callbacks=[xgb.callback.EarlyStopping(50)])
            min_mae.append(min(list(evals_result["test"].values())[0]))
            return np.array(min_mae).mean()


def xgb_objective(trial, data):
    params = {
        'subsample': trial.suggest_float("subsample", 0.2, 0.8),
        'eta': trial.suggest_float("eta", 0.01, 0.1, step=0.01),
        'max_depth': trial.suggest_int("max_depth", 1, 5, 1),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.2, 0.8, step=0.1),
        'gamma': trial.suggest_float("gamma", 0, 1),
        'eval_metric': "mae"
    }
    loss = xgb_train(params, data)

    return loss


def get_prediction(params, data1, data2):
    pre_reslust = xgb_train(params=params, data=data1, data2=data2, Predict=True)
    return pre_reslust


def XGB_tool(task: str, file_path: str):
    # check task
    if task in ("Train_only", "Prediction"):
        pass
    else:
        # 如果 task 参数不是期望的值，抛出异常或返回错误信息
        raise ValueError("The task must be Train_only or Prediction.")
    if task == "Train_only":
        new_file_path = data_preprocessing(file_path)
        data = read_excel_data(new_file_path)
        xgb_best_params, xgb_best_mae = optimizer_optuna(n_trials=10, algo="TPE", optuna_objective=xgb_objective,
                                                       data=data)
        response = f"On dataset {file_path}, after Bayesian optimization, the best hyperparameter of the random forest model is {xgb_best_params} and the best MAE metric is {xgb_best_mae}."
        return response

    elif task == "Prediction":
        train_dateset, prediction_dateset = file_path.split(",")
        train_dateset,prediction_dateset = data_preprocessing(train_dateset, prediction_dateset)
        data1 = read_excel_data(train_dateset)
        data2 = read_excel_data(prediction_dateset)
        xgb_best_params, xgb_best_mae = optimizer_optuna(n_trials=100, algo="TPE", optuna_objective=xgb_objective,
                                                         data=data1)
        pre_reslust = get_prediction(xgb_best_params, data1, data2)
        response = f"The prediction of random forest is {pre_reslust}."
        return response


class XGB_PreSchema(BaseModel):

    task: str = Field(description="Should be one of the two, Train_only or Prediction. Train_only represents that the model is only trained and returns the best hyperparameters for the XGBOOST model and the corresponding evaluation metrics. Prediction represents that the model gives predictions for the test set.")
    f_path: str = Field(description = "Should be a dataset address. When the task is Train_only, only the file address of a training set needs to be entered. When the task is Prediction, the inputs are Training Set and Test Set, separated by `,`.")


class XGB_Regression(BaseTool):
    name = "XGBOOST_Regression"
    description = "Very useful when you want to use XGBOOST algorithms."
    args_schema: Type[BaseModel] = XGB_PreSchema

    def _run(self, task: str, f_path: str) -> str:

        return XGB_tool(task, f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")