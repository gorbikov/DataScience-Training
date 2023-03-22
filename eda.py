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
    filepath = Path(str("intermediate data/" + current_script_name + "_" + filename + '_head.csv'))
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


# Выводит матрицу корреляций для датафрейма + рисует столбчатый график.
def correlation_with_target(df: pd.DataFrame, name: str, columnForCorrelation: str):
    separator_show("Матрица корреляций для " + name + " со столбцом " + columnForCorrelation)
    print(df.corr()[[columnForCorrelation]])
    # TODO Добавить столбчатую диаграмму.
    # TODO Добавить сохранение диаграммы в png.


# Выбирает столбцы типа object, выводит количество уникальных записей в каждом таком столбце.
def unique_counter_for_object_type(df: pd.DataFrame, name: str):
    separator_show("Уникальные значения в столбцах типа object в " + name)
    dummieCounter = 0
    for col in df.columns:
        if df[col].dtypes == object:
            dummieCounter += len(df[col].unique())
            print('Unique in ' + str(col) + ': ' + str(len(df[col].unique())))
    print('Dummie columns: ' + str(dummieCounter))


# Сохраняет датафрейм в виде csv в папку results.
def results_save(df: pd.DataFrame, current_script_name: str, filename: str):
    separator_show("Сохраняем датафрейм " + filename + " в csv")
    filepath = Path(str("results/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(filename + " сохранен в " + str(filepath))
