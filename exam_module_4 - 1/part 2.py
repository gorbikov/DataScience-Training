# Настраиваем импорты.
import pathlib
import pandas as pd
from eda import *

# Получаем имя текущего скрипта для сохранения выводов.
currentScriptName = pathlib.Path(__file__).name

# Вводные.
cv_path = pathlib.Path("intermediate data/results/part 1.py_cv_df.csv")
cv_target_path = pathlib.Path("intermediate data/results/part 1.py_cv_df_target.csv")
test_path = pathlib.Path("intermediate data/results/part 1.py_test_df.csv")
train_path = pathlib.Path("intermediate data/results/part 1.py_train_df.csv")
train_target_path = pathlib.Path("intermediate data/results/part 1.py_train_df_target.csv")

# Создаёт датафреймы.
cv_df = pd.read_csv(cv_path)
cv_target_df = pd.read_csv(cv_target_path)
test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)
train_target_df = pd.read_csv(train_target_path)

show_separator("6. Подумать над вариантом модели, для того чтобы решить задачу (либо ансамблем моделей)",
               size="large")
print('Так как n << m, лучше использовать логистическую регрессию, либо SMV without kernel.')

show_separator("7. Подумать нужно ли применять Unsupervised learning подход для решения задачи? "
               "Неоходима ли дополнительная информация?", size="large")
print("Применять ансупервайзд не будем.")

show_separator("8. Обучить модель и вывести валидационный скор по метрике качества.", size="large")
# TODO Нормализовать данные.
# TODO Сохранить параметры нормализации.
# TODO Собрать модель логистической регрессии.
# TODO Обучить модель
# TODO Нормализовать тестовые данные.
# TODO Получить результат.
# TODO Проверить результат (хотя бы на уровне сравнение корреляций).
