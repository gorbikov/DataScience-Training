from pathlib import Path

import matplotlib
import matplotlib.pyplot
import matplotlib.figure
import matplotlib.axes
import matplotlib.patches
import numpy as np
import pandas as pd


def show_separator(*title_texts: str, size='small'):
    """Выводит разделитель в виде штриховой линии."""
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


def save_results(current_script_name: str, df: pd.DataFrame, filename: str):
    """Сохраняет датафрейм в виде csv в папку intermediate data/results/."""
    show_separator("Сохраняем датафрейм " + filename + " в csv")
    filepath = Path(str("intermediate data/results/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(filename + " сохранен в " + str(filepath))


def inspect_data(current_script_name: str, df: pd.DataFrame, filename: str):
    """Выводит инфо, сохраняет голову датафрейма в csv в папку intermediate data/heads/."""
    show_separator("Информация по " + filename)
    df.info()
    filepath = Path(str("intermediate data/heads/" + current_script_name + "_" + filename + '_head.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.head().to_csv(filepath)
    filepath = Path(str("intermediate data/heads/" + current_script_name + "_" + filename + '_describe.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.describe().to_csv(filepath)


def search_duplicates(df: pd.DataFrame, df_name: str):
    """Выводит количество дубликатов в датафрейме."""
    show_separator("Поиск дубликатов в " + df_name)
    print("Количество дубликатов в " + df_name + ":")
    print(df[df.duplicated()].shape[0])


def delete_duplicates(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """Удаляет дубликаты в датафрейме."""
    show_separator("Удаление дубликатов в " + df_name)
    print("Размер " + df_name + " до удаления дубликатов: " + str(df.shape))
    df = df.drop_duplicates()
    print("Размер " + df_name + " после удаления дубликатов: " + str(df.shape))
    return df


def show_nans(df: pd.DataFrame, df_name: str):
    """Выводит количество пустых клеток в датафрейме."""
    show_separator("Поиск пустых клеток в " + df_name)
    print("Количество строк с пустыми клетками в " + df_name + ":")
    print(df[df.isnull().any(axis=1)].shape[0])
    print("Количество столбцов с пустыми клетками в " + df_name + ":")
    print(df.loc[:, df.isnull().any()].columns.size)


def count_unique_for_object_type(df: pd.DataFrame, df_name: str):
    """Выбирает столбцы типа object, выводит количество уникальных записей в каждом таком столбце."""
    show_separator("Уникальные значения в столбцах типа object в " + df_name)
    dummy_counter = 0
    for col in df.columns:
        if df[col].dtypes == object:
            dummy_counter += len(df[col].unique())
            print('Unique in ' + str(col) + ': ' + str(len(df[col].unique())))

    print('Dummy columns: ' + str(dummy_counter))


def generate_boxplot(current_script_name: str, df: pd.DataFrame, df_name: str, column_name: str):
    """Сохраняет график с выбросами в папку "intermediate data/diagrams/."""
    show_separator("Распределение значений для столбца " + column_name + " в датафрейме " + df_name)
    current_column = df[[column_name]]
    plt = matplotlib.pyplot()
    plt.figure(figsize=(19.2, 10.8))
    plt.boxplot(current_column)
    plt.title("Распределение значений для столбца " + column_name + " в датафрейме " + df_name)
    plt.grid()
    # TODO добавить подписи данных.
    plt.tight_layout()
    filepath = Path(
        str("intermediate data/diagrams/" + current_script_name + "_" + df_name + "_" + column_name + '_boxplot.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print("Сохранено в intermediate data/diagrams/")
    plt.close()


def generate_histogram(current_script_name: str, df: pd.DataFrame, df_name: str, column_name: str):
    """Сохраняет гистограмму в папку "intermediate data/diagrams/."""
    show_separator("Распределение значений для столбца " + column_name + " в датафрейме " + df_name)
    current_column = df[[column_name]]
    plt = matplotlib.pyplot()
    plt.figure(figsize=(19.2, 10.8))
    plt.hist(current_column, bins=100)
    plt.title("Гистограмма для столбца " + column_name + " в датафрейме " + df_name)
    plt.grid()
    # TODO добавить подписи данных.
    plt.tight_layout()
    filepath = Path(
        str("intermediate data/diagrams/" + current_script_name + "_" + df_name + "_" + column_name + '_histogram.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print("Сохранено в intermediate data/diagrams/")
    plt.close()


def generate_correlation_with_target(current_script_name: str, df: pd.DataFrame, df_name: str,
                                     column_for_correlation: str):
    """Выводит матрицу корреляций для датафрейма + сохраняет столбчатый график в папку intermediate data/diagrams/."""

    # Выводит разделитель с описанием того, что делает функция.
    show_separator("Матрица корреляций для " + df_name + " со столбцом " + column_for_correlation)

    # Считает и выводит матрицу корреляций.
    corr_df = df.corr()[[column_for_correlation]].drop(column_for_correlation, axis=0)
    print(corr_df.sort_values(by=[column_for_correlation], ascending=False))

    # Формирует данные для составления графика.
    plt_labels = corr_df.index.values
    plt_values = corr_df[[column_for_correlation]].values.reshape(plt_labels.shape[0])

    # Создаёт фигуру, оси и решетку.
    fig: matplotlib.figure.Figure = matplotlib.pyplot.figure()
    ax: matplotlib.axes.Axes = matplotlib.pyplot.axes()
    fig.set_size_inches(w=19.2, h=10.8)
    ax.grid()

    # Рисует график.
    ax.set_title("Матрица корреляций для " + df_name + " со столбцом " + column_for_correlation)
    y_pos = np.arange(len(plt_labels))
    ax.barh(y_pos, plt_values)
    ax.set_yticks(y_pos, plt_labels)

    # Добавляет подписи данных.
    graph_patches = ax.patches
    for patch, value in zip(graph_patches, plt_values):
        patch: matplotlib.patches.Rectangle = patch
        width = patch.get_width()
        ax.text(width/2, patch.get_y()+patch.get_height()/2, round(value, 2), verticalalignment='center',
                bbox={'facecolor': 'grey','edgecolor':'None', 'alpha': 0.5, 'pad': 0.3})

    # Регулирует отступы на графике.
    matplotlib.pyplot.tight_layout()

    # Сохраняет график в файл.
    filepath = Path(str("intermediate data/diagrams/" + current_script_name + "_" + df_name + '_correlation.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.pyplot.savefig(filepath)

    # Убирает фигуру из памяти и закрывает график.
    fig.clear()
    matplotlib.pyplot.close()
