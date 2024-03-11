from openfe import OpenFE, transform
# import sys
# sys.path.append('../')
import pandas as pd
from openfe import OpenFE, tree_to_formula, transform
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import statsmodels.api as sm
from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
import os


def get_score(train_x, test_x, train_y, test_y):
    n_jobs = 4
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2020)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    return score,test_y, pred


def data_generator(f_path, seed=42):
    df_all = pd.read_excel(f_path)
    # create a min max processing object
    composition = df_all
    scaler = preprocessing.StandardScaler().fit(composition)
    normalized_composition = scaler.transform(composition)

    return normalized_composition


def auto_fe(data):
    origin_data = data.copy(deep = True)
    col = data.columns
    n_jobs = 4
    label = data[col[-1]]
    del data[col[-1]]
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
    # get baseline score
    score,_,_ = get_score(train_x, test_x, train_y, test_y)
    print("The MSE before feature generation is", score)
    # feature generation
    ofe = OpenFE()
    ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs,n_data_blocks = 1)
    # OpenFE recommends a list of new features. We include the top 10
    # generated features to see how they influence the model performance
    train_x, test_x = transform(train_x, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
    score,test_y, pred = get_score(train_x, test_x, train_y, test_y)
    print("The MSE after feature generation is", score)
    print("The top 10 generated features are")
    for feature in ofe.new_features_list[:10]:
        print(tree_to_formula(feature))
    new_feature_name = []
    for i in list(origin_data.columns)[:-1]:
        new_feature_name.append(i)
    for feature in ofe.new_features_list[:10]:
        new_feature_name.append(tree_to_formula(feature))
    new_feature_name.append(list(origin_data.columns)[-1])
    train_data = pd.concat([train_x, train_y], axis=1)
    test_data = pd.concat([test_x, test_y], axis=1)
    data_concat = pd.concat([train_data, test_data], axis=0)
    data_concat.columns = new_feature_name
    data_concat.to_excel("特征生成.xlsx", index=False)
    return data_concat


def Liner_model(datasets,cols):
    skf = KFold(n_splits=5, shuffle=True,random_state=888)
    valid_ssr =[]
    x = datasets[:,cols]
    y = datasets[:,-1:]
    for i, (trn_idx, val_idx) in enumerate(skf.split(x, y)):
        trn_x, trn_y = x[trn_idx], y[trn_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        # 创建并拟合线性回归模型
        model = LinearRegression()
        model.fit(trn_x, trn_y)
        # 获得残差和
        residual_sum_of_squares = np.sum((model.predict(val_x) - val_y) ** 2)
        valid_ssr.append(residual_sum_of_squares)
    #返回当前特征组合的交叉验证的残差和
    return np.array(valid_ssr).mean()


def bic(data, features):
    # 估计线性回归模型的参数和残差平方和
    rss = Liner_model(data, features)
    # 计算BIC值
    n = len(data) * 0.8
    k = len(features)

    return n * np.log(rss / n) + k * np.log(n), rss


def get_best_features(data):
    # 初始化特征集合为空
    best_features = []
    current_best_bic = []
    current_best_feature = []
    SSE = []
    for a in range(data.shape[1] - 1):
        # 初始化最小的BIC值为无穷大
        Bics = []
        cols = []
        sses = []
        # 遍历所有的特征
        for i in range(data.shape[1] - 1):
            col = current_best_feature.copy()
            if i not in col:
                col.append(i)
                # 计算添加后的特征子集的BIC值
                new_bic,sse = bic(data, col)
                Bics.append(new_bic)
                cols.append(col)
                sses.append(sse)
        current_best_feature = cols[np.array(sses).argmin()].copy()
        current_best_bic.append(Bics[np.array(Bics).argmin()])
        best_features.append(current_best_feature)
        SSE.append(sses[np.array(Bics).argmin()])
    return best_features,SSE


def get_bis(features,path):
    org_data = pd.read_excel(path).fillna(0)
    columns = list(org_data.columns)
    data_y = org_data[[columns[i] for i in features]]
    model_1 = sm.OLS(org_data['average_coulombic_efficiency'], sm.add_constant(data_y)).fit()
    return model_1.bic


def bic_choose_feature(best_features,path):
    bics = []
    for i in range(len(best_features)):
        bic = get_bis(best_features[i],path)
        bics.append(bic)
    min_index = np.array(bics).argmin()
    return best_features[min_index],bics


def Automated_Feature_Engineering_and_Screening(path):
    data = read_excel_data(path).fillna(0)
    best_features,SSE = get_best_features(np.array(data))
    pd.DataFrame(best_features).to_excel("best_feature_candinate.xlsx")
    bic_feature,bics = bic_choose_feature(best_features,path)
    sse_bics = pd.concat((pd.DataFrame(SSE),pd.DataFrame(bics)),axis=1)
    sse_bics.columns = ["SSE","BIC"]
    sse_bics.to_excel("SSE_and_BIC.xlsx",index = False)
    org_data = pd.read_excel("Feature_Generation.xlsx").fillna(0)
    columns = list(org_data.columns)
    filter_data = org_data[[columns[i] for i in bic_feature]]
    data3 = pd.concat((filter_data,org_data.iloc[:,-1:]),axis = 1)
    data3.to_excel("data_after_BIC.xlsx",index = False)
    save_path = os.getcwd() + "\\data_after_BIC.xlsx"
    return f"Automated feature engineering has been completed and the features filtered by the Bayesian information criterion are saved on {save_path}."