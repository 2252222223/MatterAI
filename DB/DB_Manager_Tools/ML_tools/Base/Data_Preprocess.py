from DB.DB_Manager_Tools.ML_tools.Base.Data_loder import read_excel_data
from sklearn import preprocessing
import pandas as pd
from typing import Optional, Type, Union, Tuple


def data_preprocessing(f_path: str, test_path: Optional[str] = None, seed=42):
    data = read_excel_data(f_path)
    scaler = preprocessing.StandardScaler().fit(data)
    data_tra = scaler.transform(data)
    new_data = pd.DataFrame(data_tra, columns=data.columns)
    new_path = f_path.replace(".xls", "").replace(".csv", "") + "_transform.csv"
    new_data.to_csv(new_path, index=False)
    if test_path is not None:
        test_data = read_excel_data(test_path)
        new_scaler = preprocessing.StandardScaler().fit(data.iloc[:, :-1])
        test_data_tra = new_scaler.transform(test_data)
        new_test_data = pd.DataFrame(test_data_tra, columns=test_data.columns)
        new_test_path = test_path.replace(".xls", "").replace(".csv", "").replace(".xlsx", "") + "_transform.csv"
        new_test_data.to_csv(new_test_path, index=False)
        return new_path,new_test_path

    else:
        return new_path
