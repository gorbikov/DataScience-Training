from pathlib import Path
import pandas as pd


# Выводит разделитель в виде штриховой линии.
def separator_show(*args: str, type='small'):
    match type:
        case "small":
            print()
            for ar in args:
                print(ar)
            print("---------------------------------------------------------------------")
        case "large":
            print()
            print()
            for ar in args:
                print(ar)
            print("===================================================================================================")


# Выводит инфо, сохраняет голову датафрейма в csv в папку intermediate data.
def data_inspection(df: pd.DataFrame, current_script_name: str, filename: str):
    separator_show("Информация по " + filename)
    df.info()
    filepath = Path(str("intermediate data/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.head().to_csv(filepath)


# Выводит количество дубликатов в датафрейме.
def duplicates_search(df: pd.DataFrame, name: str):
    separator_show("Поиск дубликатов в " + name)
    print("Количество дубликатов в " + name + ":")
    print(df[df.duplicated()].shape[0])


# Удаляет дубликаты в датафрейме.
def duplicates_delete(df: pd.DataFrame, name: str):
    separator_show("Удаление дубликатов в " + name)
    print("Размер " + name + " до удаления дубликатов: " + str(df.shape))
    df = df.drop_duplicates()
    print("Размер " + name + " после удаления дубликатов: " + str(df.shape))
    return df


# Сохраняет датафрейм в виде csv в папку results.
def intermediate_data_save(df: pd.DataFrame, current_script_name: str, filename: str):
    filepath = Path(str("results/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
