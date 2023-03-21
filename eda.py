import pandas as pd


# Выводит инфо, сохраняет голову датафрейма в csv в папку intermediate data.
def data_inspection(df: pd.DataFrame, current_script_name: str, filename: str):
    df.info()
    df.head().to_csv("intermediate data/" + current_script_name + "_" + filename)


def duplicate_search(df: pd.DataFrame, name: str):
    print("Количество дубликатов в " + name + ":")
    print(df[df.duplicated()].shape[0])
