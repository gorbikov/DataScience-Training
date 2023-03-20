import pandas as pd


def prosmotr_dannih(df: pd.DataFrame, current_script_name: str, filename: str):
    df.info()
    df.head().to_csv("intermediate data/" + current_script_name + "_" + filename)

def poisk_dublikatov(df: pd.DataFrame):
    print(df[df.duplicated()].shape)