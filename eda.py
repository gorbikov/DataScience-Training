from pathlib import Path

import matplotlib.pyplot
import pandas as pd


def separator_show(*title_texts: str, size='small'):
    # Выводит разделитель в виде штриховой линии.
    match size:
        case "small":
            print()
            for title_text in title_texts:
                print(title_text)
            print("---------------------------------------------------------------------")
        case "large":
            print()
            print()
            for title_text in title_texts:
                print(title_text)
            print("===================================================================================================")


def results_save(df: pd.DataFrame, current_script_name: str, filename: str):
    # Сохраняет датафрейм в виде csv в папку intermediate data/results/.
    separator_show("Сохраняем датафрейм " + filename + " в csv")
    filepath = Path(str("intermediate data/results/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(filename + " сохранен в " + str(filepath))


def data_inspection(df: pd.DataFrame, current_script_name: str, filename: str):
    # Выводит инфо, сохраняет голову датафрейма в csv в папку intermediate data/heads/.
    separator_show("Информация по " + filename)
    df.info()
    filepath = Path(str("intermediate data/heads/" + current_script_name + "_" + filename + '_head.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.head().to_csv(filepath)


def duplicates_search(df: pd.DataFrame, df_name: str):
    # Выводит количество дубликатов в датафрейме.
    separator_show("Поиск дубликатов в " + df_name)
    print("Количество дубликатов в " + df_name + ":")
    print(df[df.duplicated()].shape[0])


def duplicates_delete(df: pd.DataFrame, df_name: str):
    # Удаляет дубликаты в датафрейме.
    separator_show("Удаление дубликатов в " + df_name)
    print("Размер " + df_name + " до удаления дубликатов: " + str(df.shape))
    df = df.drop_duplicates()
    print("Размер " + df_name + " после удаления дубликатов: " + str(df.shape))
    return df


def unique_counter_for_object_type(df: pd.DataFrame, df_name: str):
    # Выбирает столбцы типа object, выводит количество уникальных записей в каждом таком столбце.
    separator_show("Уникальные значения в столбцах типа object в " + df_name)
    dummy_counter = 0
    for col in df.columns:
        if df[col].dtypes == object:
            dummy_counter += len(df[col].unique())
            print('Unique in ' + str(col) + ': ' + str(len(df[col].unique())))
    print('Dummy columns: ' + str(dummy_counter))


def correlation_with_target(df: pd.DataFrame, df_name: str, column_for_correlation: str, current_script_name: str):
    # Выводит матрицу корреляций для датафрейма + рисует столбчатый график, сохраняет в папку intermediate data/diagrams/.
    separator_show("Матрица корреляций для " + df_name + " со столбцом " + column_for_correlation)
    corr_df = df.corr()[[column_for_correlation]].drop(column_for_correlation, axis=0)
    print(corr_df.sort_values(by=[column_for_correlation], ascending=False))
    x_axis = corr_df.index.values
    y_axis = corr_df[[column_for_correlation]].values

    # TODO Передалать графики на fig. Добавить тайтл.
    fig = matplotlib.pyplot.bar(x_axis, y_axis)

    # df.corr()[[column_for_correlation]].drop(column_for_correlation, axis=0) \
    #     .plot(y=[column_for_correlation], kind="bar")
    matplotlib.pyplot.show()

    filepath = Path(str("intermediate data/diagrams/" + current_script_name + "_" + df_name + '.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.pyplot.savefig(filepath)


def numerical_anomaly_show(df: pd.DataFrame, df_name: str, *column_names: str):
    for column_name in column_names:
        separator_show("Распределение значений для столбца " + column_name + " в датафрейме " + df_name)
        current_column = df[[column_name]]
        matplotlib.pyplot.boxplot(current_column)
        matplotlib.pyplot.show()
        # TODO Передалать графики на fig. Добавить тайтл. Добавить сохранение.
