# Настраиваем импорты.
import pathlib

from eda import *

# Получаем имя текущего скрипта для сохранения выводов.
currentScriptName = pathlib.Path(__file__).name

# Вводные.
train_path = pathlib.Path('data/exam_module_4/1_variant_dna_sequence_mutation_prediction/input/train.csv')
test_path = pathlib.Path('data/exam_module_4/1_variant_dna_sequence_mutation_prediction/input/test.csv')
cv_fraction = 0.15
# Фиксируем рандом.
random_ceed = 777

show_separator('1. Определяем тип задачи.', size="large")
print('Task type: logistic regression or SVM')

show_separator("2. Создаём фреймы и выделяем часть датасета на CV.", size="large")
# Формируем оригинальные датафреймы
original_train_cv_df = pd.read_csv(train_path, index_col='ID')
original_test_df = pd.read_csv(test_path, index_col='ID')

show_separator("3. Определить тип переменных в датасете.", size="large")
# Смотрим оригинальные датафреймы.
inspect_data(currentScriptName, original_train_cv_df, "original_train_cv_df")
inspect_data(currentScriptName, original_test_df, "original_test_df")

# Ищем и удаляем дубликаты в оригинальных данных.
search_duplicates(original_train_cv_df, "original_train_cv_df")
search_duplicates(original_test_df, "original_test_df")

# Ищем незаполненные и некорректно заполненные данные.
show_nans(original_train_cv_df, "original_train_cv_df")
show_nans(original_test_df, "original_test_df")

# Ищем аномалии.
columns = original_train_cv_df.iloc[:, 9:19].columns.tolist()
for column in columns:
    generate_histogram(currentScriptName, original_train_cv_df, "original_train_cv_df", column, bins=50)

columns = original_train_cv_df.iloc[:, 30:31].columns.tolist()
for column in columns:
    generate_histogram(currentScriptName, original_train_cv_df, "original_train_cv_df", column, bins=2)

columns = original_train_cv_df.iloc[:, 19:30].columns.tolist()
for column in columns:
    generate_boxplot(currentScriptName, original_train_cv_df, "original_train_cv_df", column)
    generate_histogram(currentScriptName, original_train_cv_df, "original_train_cv_df", column, bins=100)

show_separator("""4. Если это необходимо провести препроцессинг данных, нужно ли применять алгоритмы понижения
размерности? Нужно ли убирать аномалии?""", size="large")

# Переводим первые 9 столбцов в цифры (по методу one-hot, one-hot столбцы добавляются в конце датафрейма).
count_unique_for_object_type(original_train_cv_df, "original_train_cv_df")
count_unique_for_object_type(original_test_df, "original_test_df")
train_cv_df = pd.get_dummies(original_train_cv_df)
test_df = pd.get_dummies(original_test_df)

show_separator("""5. Провести EDA и вывести какие-то умозаключения и посмотреть на распределения признаков, на
корреляции, на выбросы.""", size='large')

generate_correlation_with_target(currentScriptName, train_cv_df, "train_cv_df", "mutation")

# Формируем датафреймы из псевдослучайных выборок.
train_df = train_cv_df.sample(frac=(1 - cv_fraction), random_state=random_ceed).drop('mutation', axis=1)
train_df_target = train_cv_df.sample(frac=(1 - cv_fraction), random_state=random_ceed)[['mutation']]
cv_df = train_cv_df.drop(train_df.index).drop('mutation', axis=1)
cv_df_target = train_cv_df.drop(train_df_target.index)[['mutation']]

# Просмотр данных перед сохранением.
inspect_data(currentScriptName, train_df, "train_df")
inspect_data(currentScriptName, train_df_target, "train_df_target")
inspect_data(currentScriptName, cv_df, "cv_df")
inspect_data(currentScriptName, cv_df_target, "cv_df_target")
inspect_data(currentScriptName, test_df, "test_df")

# Сохраняем данные в csv.
save_results(currentScriptName, train_df, "train_df")
save_results(currentScriptName, train_df_target, "train_df_target")
save_results(currentScriptName, cv_df, "cv_df")
save_results(currentScriptName, cv_df_target, "cv_df_target")
save_results(currentScriptName, test_df, "test_df")
