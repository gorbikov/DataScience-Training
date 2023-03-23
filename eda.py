from pathlib import Path

import matplotlib.pyplot
import pandas as pd


def separator_show(*args: str, size='small'):
    # Выводит разделитель в виде штриховой линии.
    match size:
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


def data_inspection(df: pd.DataFrame, current_script_name: str, filename: str):
    # Выводит инфо, сохраняет голову датафрейма в csv в папку intermediate data/heads/.
    separator_show("Информация по " + filename)
    df.info()
    filepath = Path(str("intermediate data/heads/" + current_script_name + "_" + filename + '_head.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.head().to_csv(filepath)


def duplicates_search(df: pd.DataFrame, name: str):
    # Выводит количество дубликатов в датафрейме.
    separator_show("Поиск дубликатов в " + name)
    print("Количество дубликатов в " + name + ":")
    print(df[df.duplicated()].shape[0])


def duplicates_delete(df: pd.DataFrame, name: str):
    # Удаляет дубликаты в датафрейме.
    separator_show("Удаление дубликатов в " + name)
    print("Размер " + name + " до удаления дубликатов: " + str(df.shape))
    df = df.drop_duplicates()
    print("Размер " + name + " после удаления дубликатов: " + str(df.shape))
    return df


def unique_counter_for_object_type(df: pd.DataFrame, name: str):
    # Выбирает столбцы типа object, выводит количество уникальных записей в каждом таком столбце.
    separator_show("Уникальные значения в столбцах типа object в " + name)
    dummy_counter = 0
    for col in df.columns:
        if df[col].dtypes == object:
            dummy_counter += len(df[col].unique())
            print('Unique in ' + str(col) + ': ' + str(len(df[col].unique())))
    print('Dummy columns: ' + str(dummy_counter))


def correlation_with_target(df: pd.DataFrame, name: str, column_for_correlation: str, current_script_name: str):
    # Выводит матрицу корреляций для датафрейма + рисует столбчатый график, сохраняет в папку intermediate data/diagrams/.
    separator_show("Матрица корреляций для " + name + " со столбцом " + column_for_correlation)
    print(df.corr()[[column_for_correlation]].drop(column_for_correlation, axis=0)
          .sort_values(by=[column_for_correlation], ascending=False))
    df.corr()[[column_for_correlation]].drop(column_for_correlation, axis=0) \
        .plot(y=[column_for_correlation], kind="bar")
    filepath = Path(str("intermediate data/diagrams/" + current_script_name + "_" + name + '.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.pyplot.savefig(filepath)
    matplotlib.pyplot.show()


def results_save(df: pd.DataFrame, current_script_name: str, filename: str):
    # Сохраняет датафрейм в виде csv в папку intermediate data/results/.
    separator_show("Сохраняем датафрейм " + filename + " в csv")
    filepath = Path(str("intermediate data/results/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(filename + " сохранен в " + str(filepath))
