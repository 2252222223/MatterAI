import torch
import torch.nn as nn
import time
import copy
import numpy as np
from sklearn.model_selection import KFold
import torch.optim as optim
from DB.DB_Manager_Tools.ML_tools.Base.Optimizer import optimizer_optuna
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
from DB.DB_Manager_Tools.ML_tools.Base.Data_Preprocess import data_preprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 搭建全连接神经网络回归
class MLPregression(nn.Module):
    def __init__(self, params, data):
        super(MLPregression, self).__init__()
        self.layber_number = params["layber_number"]
        self.unit = params["unit"]
        # 第一个隐含层
        self.hidden1 = nn.Linear(in_features=data.shape[-1]-1, out_features=self.unit, bias=True)
        # 第二个隐含层
        self.hidden2 = nn.Linear(self.unit, self.unit)
        # 第三个隐含层
        self.hidden3 = nn.Linear(128, 256)
        # 回归预测层
        self.hidden5 = nn.Linear(self.unit, 64)
        self.predict = nn.Linear(64, 1)
        self.relu = nn.functional.relu
        self.dropout = nn.Dropout(params["drop_out"])

    # 定义网络前向传播路径
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        for i in range(self.layber_number):
            x = self.dropout(self.relu(self.hidden2(x)))
        x = self.dropout(self.relu(self.hidden5(x)))
        output = self.predict(x)
        # 输出一个一维向量
        return output[:, 0]


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def train_model(params, model, dataloaders, criterion, optimizer, num_epochs, i, savebest=False):
    since = time.time()
    best_loss = 10000000
    model.to(device)
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #                     print(outputs.shape, labels.shape)
                    loss = criterion(outputs, labels)
                    #                     print("loss为：",loss)
                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #                 计算损失

                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataloaders[phase].dataset_len
            #             print(epoch_loss)
            time_elapsed = time.time() - since
            #             print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            #             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                if savebest is True:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    state = {
                        'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict(),  # 优化器的状态信息
                    }
                    filename = './NN_best' + str(i) + '.pth'
                    torch.save(state, filename)
            if phase == 'valid':
                valid_losses.append(epoch_loss)
            #                 scheduler.step(epoch_loss)#学习率衰减
            if phase == 'train':
                train_losses.append(epoch_loss)

    if savebest is True:
        return train_losses, valid_losses, best_loss
    else:
        return best_loss


def model_cross_train(params, data, savebest=False):
    x = data[:, :-1]
    y = data[:, -1:]

    loss = []
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_losses = []
    valid_losses = []

    for i, (trn_idx, val_idx) in enumerate(skf.split(x, y)):
        trn_x, trn_y = x[trn_idx], y[trn_idx]
        val_x, val_y = x[val_idx], y[val_idx]

        # 将数据集转为张量
        X_train_t = torch.from_numpy(trn_x.astype(np.float32))
        y_train_t = torch.from_numpy(trn_y.squeeze().astype(np.float32))
        X_valid_t = torch.from_numpy(val_x.astype(np.float32))
        y_valid_t = torch.from_numpy(val_y.squeeze().astype(np.float32))

        # 将训练数据处理为数据加载器
        #         train_data = Data.TensorDataset(X_train_t, y_train_t)
        #         valid_data = Data.TensorDataset(X_valid_t, y_valid_t)

        #         train_loader = Data.DataLoader(dataset = train_data, batch_size = params['batch_size'],
        #                                        shuffle = True, num_workers = 1)
        #         val_loader = Data.DataLoader(dataset = valid_data, batch_size = params['batch_size'],
        #                                        shuffle = True, num_workers = 1)
        train_loader = FastTensorDataLoader(X_train_t, y_train_t, batch_size=params['batch_size'],
                                            shuffle=True)
        val_loader = FastTensorDataLoader(X_valid_t, y_valid_t, batch_size=params['batch_size'],
                                          shuffle=True)

        # 实例化
        model = MLPregression(params, data)
        # 损失函数
        criterion = torch.nn.L1Loss()
        # 优化器设置
        optimizer_ft = optim.Adam(model.parameters(), lr=params["lr"])
        # scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#学习率每7个epoch衰减成原来的1/1
        # 数据加载器
        dataloaders = {'train': train_loader, 'valid': val_loader}

        if savebest is True:
            train_loss, valid_loss, best_loss = train_model(params, model, dataloaders, criterion, optimizer_ft,
                                                            num_epochs=100, i=i, savebest=True)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            loss.append(best_loss)
        else:
            best_loss = train_model(params, model, dataloaders, criterion, optimizer_ft, num_epochs=20, i=i)
            loss.append(best_loss)

    if savebest is True:
        return train_losses, valid_losses, loss
    else:
        return np.array(loss).mean()


def model_objective(trial, data):
    params = {
          'lr': trial.suggest_float("lr", 1e-5, 1e-2),
          'batch_size': trial.suggest_int("batch_size", 16,128,16),
          'drop_out':trial.suggest_float("drop_out", 0, 0.2),
          'unit': trial.suggest_int("unit", 16, 128,16),
          'layber_number':trial.suggest_int("layber_number", 1, 16,1)
          }
    loss = model_cross_train(params, data)
    return loss


def get_prediction(model_best_params,test_data):
    test_x = test_data
    # 将数据集转为张量
    X_test_t = torch.from_numpy(test_x.astype(np.float32))
    X_test_t = X_test_t.to(device)
    y_pre_all = 0
    for i in range(10):
        #     最好模型加载
        best_model = MLPregression(model_best_params)
        path = './NN_best' + str(i) + '.pth'
        chickpoint = torch.load(path)
        best_model.load_state_dict(chickpoint["state_dict"])
        best_model.to(device)

        y_pre = best_model(X_test_t)
        y_pre = y_pre.cpu()
        y_pre = y_pre.data.numpy()
        y_pre_all += y_pre/10

    return y_pre_all


def NN_tool(task: str, file_path: str):
    # check task
    if task in ("Train_only", "Prediction"):
        pass
    else:
        # 如果 task 参数不是期望的值，抛出异常或返回错误信息
        raise ValueError("The task must be Train_only or Prediction.")

    if task == "Train_only":

        new_path = data_preprocessing(file_path)
        dataset = read_excel_data(new_path)

        data = np.array(dataset)
        nn_best_params, nn_best_mae = optimizer_optuna(n_trials=3, algo="TPE", optuna_objective=model_objective,
                                                       data=data)
        response = f"On dataset {file_path}, after Bayesian optimization, the best hyperparameter of the random forest model is {nn_best_params} and the best MAE metric is {nn_best_mae}."
        return response
    elif task == "Prediction":
        train_dateset, prediction_dateset = file_path.split(",")
        new_path, new_pre_path = data_preprocessing(train_dateset, prediction_dateset)

        data1 = read_excel_data(new_path)
        data2 = read_excel_data(new_pre_path)
        nn_best_params, nn_best_mae = optimizer_optuna(n_trials=30, algo="TPE", optuna_objective=model_objective,
                                                       data=data1)
        model_cross_train(nn_best_params, data1, savebest=True)
        test_pre = get_prediction(nn_best_params, data2)
        response = f"The prediction of neural network is {test_pre}."
        return response


class NN_PreSchema(BaseModel):

    task: str = Field(description="Should be one of the two, Train_only or Prediction. Train_only represents that the model is only trained and returns the best hyperparameters for the neural network model and the corresponding evaluation metrics. Prediction represents that the model gives predictions for the test set.")
    f_path: str = Field(description = "Should be a dataset address. When the task is Train_only, only the file address of a training set needs to be entered. When the task is Prediction, the inputs are Training Set and Test Set, separated by `,`.")


class NN_Regression(BaseTool):
    name = "Neural_Network_Regression"
    description = "Very useful when you want to use neural network regression algorithms."
    args_schema: Type[BaseModel] = NN_PreSchema

    def _run(self, task: str, f_path: str) -> str:

        return NN_tool(task, f_path)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")