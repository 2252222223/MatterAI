import torch
import gpytorch
import numpy as np
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.ML_tools.Base.Data_Preprocess import data_preprocessing
from sklearn.model_selection import KFold
# 首先定义一个高斯过程回归模型
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def read_init_data(know_piont_filename, unknow_piont_filename):


    data = read_excel_data(know_piont_filename)

    X_init = torch.from_numpy(np.array(data)[:, 1:-1] if 'Unnamed: 0' in data.columns else np.array(data)[:, :-1])
    Y_init = torch.from_numpy(np.array(data)[:, -1:].reshape(-1))
    data2 = read_excel_data(unknow_piont_filename)
    x_test = torch.from_numpy(np.array(data2)[:, 1:] if 'Unnamed: 0' in data2.columns else np.array(data2)[:, :])
    print(x_test.shape)
    return X_init, Y_init, x_test


def gp_model_train(X_init, Y_init):
    # 利用初始值初始化模型与似然函数
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_init, Y_init, likelihood)
    # Optimize the hyperparameters using maximum likelihood
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(500):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_init)
        # Calc loss and backprop gradients
        loss = -mll(output, Y_init).mean()
        loss.backward()
        if i % 5 == 4:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, 100, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    return model, likelihood


def model_eval(model, likelihood, x_test):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        observed_pred = likelihood(model(x_test))
        pred_mean = observed_pred.mean
        pred_std = observed_pred.stddev
    return observed_pred, pred_mean, pred_std


# 定义标准正态分布的累积分布函数和概率密度函数
cdf = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0)).cdf
pdf = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0)).log_prob


# 定义 EI 采集函数
def expected_improvement(pred_mean, pred_std, best_f, sigma):
    """
    EI 采集函数的实现
    """
    mu, std = pred_mean, pred_std
    # 将预测方差中小于等于0的部分替换为很小的正数，防止出现无穷大或 NaN
    std = torch.max(std, torch.tensor(1e-9, dtype=torch.float))

    # 计算标准正态分布的累积分布函数和概率密度函数
    t = (mu - best_f - sigma) / std
    ei = (mu - best_f) * cdf(t) + std * np.exp(pdf(t))
    # 将 EI 值中小于等于0的部分替换为很小的正数，防止出现无穷大或 NaN
    ei = torch.max(ei, torch.tensor(1e-9, dtype=torch.float))
    return ei


def search_next_point(Y_init, pred_mean, pred_std, x_test):
    # 计算所有未知点的EI
    best_f = Y_init.max()
    ei = expected_improvement(pred_mean, pred_std, best_f, sigma=0)
    # 得到EI最大值的索引
    max_index = ei.argmax()
    # 得到最有价值的下一个需要测试的点，保留两位小数
    x_next = x_test[max_index]
    y_next = pred_mean[max_index]
    return x_next, y_next


def GP_only_train(path):
    train_dateset_path = data_preprocessing(path)
    data = data_preprocessing(train_dateset_path)
    x = data[:, :-1]
    y = data[:, -1:]
    criterion = torch.nn.L1Loss()
    loss = []
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (trn_idx, val_idx) in enumerate(skf.split(x, y)):
        trn_x, trn_y = x[trn_idx], y[trn_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        # 将数据集转为张量
        X_train_t = torch.from_numpy(trn_x.astype(np.float32))
        y_train_t = torch.from_numpy(trn_y.squeeze().astype(np.float32))
        X_valid_t = torch.from_numpy(val_x.astype(np.float32))
        y_valid_t = torch.from_numpy(val_y.squeeze().astype(np.float32))
        model, likelihood = gp_model_train(X_train_t, y_train_t)
        observed_pred, pred_mean, pred_std = model_eval(model, likelihood, X_valid_t)
        mae = criterion(observed_pred, y_valid_t)
        loss.append(mae)
    mae_mean = np.array(loss).mean()
    return f"On dataset {path}, and the best MAE metric is {mae_mean} for Gaussian process Regression."

def gp_activate_learning(task, f_path):
    # check task
    if task in ("Train_only", "Active_Learning"):
        pass
    else:
        # 如果 task 参数不是期望的值，抛出异常或返回错误信息
        raise ValueError("The task must be Train_only or Prediction.")

    if task =="Active_Learning":
        train_dateset, prediction_dateset = f_path.split(",")
        know_piont_filename, unknow_piont_filename = data_preprocessing(train_dateset, prediction_dateset)
        X_init, Y_init, x_test = read_init_data(know_piont_filename,unknow_piont_filename)
        print(x_test.shape)
        model, likelihood = gp_model_train(X_init, Y_init)
        observed_pred, pred_mean, pred_std=model_eval(model, likelihood, x_test)
        x_next, y_next = search_next_point(Y_init, pred_mean, pred_std, x_test)
        return f"The next most valuable point is {x_next.numpy()}, whose referenced predicted value is {y_next.numpy()}."

    elif task =="Train_only":
        return GP_only_train(f_path)



class GP_PreSchema(BaseModel):
    task: str = Field(description="Should be one of the two, Train_only or Active_Learning. Train_only represents that the model is only trained and returns the best hyperparameters for the Gaussian process regression and the corresponding evaluation metrics. Active_Learning represents the most valuable points or components that the model explores or predicts from the test data.")
    f_path: str = Field(description = "Should be a dataset address. When the task is Train_only, only the file address of a training set needs to be entered. When the task is Active_Learning, the inputs are Training Set and Test Set, separated by `,`.")


class GP_Regression(BaseTool):
    name = "Gaussian_process_Regression"
    description = "Very useful when you need to explore or predict the most valuable point or component."
    args_schema: Type[BaseModel] = GP_PreSchema

    def _run(self, task: str, f_path: str) -> str:

        return gp_activate_learning(task, f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

