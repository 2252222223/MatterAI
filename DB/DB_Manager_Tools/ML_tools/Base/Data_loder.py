import pandas as pd


def read_excel_data(f_path):
    if f_path.endswith(".csv"):
        data = pd.read_csv(f_path)
    elif f_path.endswith(".xls") or f_path.endswith(".xlsx"):
        data = pd.read_excel(f_path)
    return data
