# Настраиваем импорты.
import pathlib

from eda import *

# Получаем имя текущего скрипта для сохранения выводов.
currentScriptName = pathlib.Path(__file__).name

# Вводные.
trainPath = pathlib.Path('data/exam_module_4/1_variant_dna_sequence_mutation_prediction/input/train.csv')
testPath = pathlib.Path('data/exam_module_4/1_variant_dna_sequence_mutation_prediction/input/test.csv')
cvFraction = 0.15
# Фиксируем рандом.
randomCeed = 777

show_separator('1. Определяем тип задачи.', size="large")
print('Task type: logistic regression or SVM')

show_separator("2. Создаём фреймы и выделяем часть датасета на CV.", size="large")
# Формируем оригинальные датафреймы
originalTrainCvDf = pd.read_csv(trainPath, index_col='ID')
originalTestDf = pd.read_csv(testPath, index_col='ID')

show_separator("3. Определить тип переменных в датасете.", size="large")
# Смотрим оригинальные датафреймы.
inspect_data(currentScriptName, originalTrainCvDf, "originalTrainCvDf")
inspect_data(currentScriptName, originalTestDf, "originalTestDf")

# Ищем и удаляем дубликаты в оригинальных данных.
search_duplicates(originalTrainCvDf, "originalTrainCvDf")
search_duplicates(originalTestDf, "originalTestDf")

# Ищем незаполненные и некорректно заполненные данные.
show_nans(originalTrainCvDf, "originalTrainCvDf")
show_nans(originalTestDf, "originalTestDf")

# Ищем аномалии.
columns = originalTrainCvDf.iloc[:, 9:19].columns.tolist()
for column in columns:
    show_histogram(currentScriptName, originalTrainCvDf, "originalTrainCvDf", False, column)

columns = originalTrainCvDf.iloc[:, 30:31].columns.tolist()
for column in columns:
    show_histogram(currentScriptName, originalTrainCvDf, "originalTrainCvDf", False, column)

columns = originalTrainCvDf.iloc[:, 19:30].columns.tolist()
for column in columns:
    show_boxplot(currentScriptName, originalTrainCvDf, "originalTrainCvDf", False, column)
    show_histogram(currentScriptName, originalTrainCvDf, "originalTrainCvDf", False, column)

show_separator("""4. Если это необходимо провести препроцессинг данных, нужно ли применять алгоритмы понижения
размерности? Нужно ли убирать аномалии?""", size="large")

# Переводим первые 9 столбцов в цифры (по методу one-hot, one-hot столбцы добавляются в конце датафрейма).
count_unique_for_object_type(originalTrainCvDf, "originalTrainCVDf")
count_unique_for_object_type(originalTestDf, "originalTestDf")
trainCvDf = pd.get_dummies(originalTrainCvDf)
testDf = pd.get_dummies(originalTestDf)

show_separator("""5. Провести EDA и вывести какие-то умозаключения и посмотреть на распределения признаков, на
корреляции, на выбросы.""", size='large')

show_correlation_with_target(currentScriptName, trainCvDf, "trainCvDf", "mutation", False)

# Формируем датафреймы из псевдослучайных выборок.
trainDf = trainCvDf.sample(frac=(1 - cvFraction), random_state=randomCeed).drop('mutation', axis=1)
trainDfTarget = trainCvDf.sample(frac=(1 - cvFraction), random_state=randomCeed)[['mutation']]
cvDf = trainCvDf.drop(trainDf.index).drop('mutation', axis=1)
cvDfTarget = trainCvDf.drop(trainDfTarget.index)[['mutation']]

# Просмотр данных перед сохранением.
inspect_data(currentScriptName, trainDf, "trainDf")
inspect_data(currentScriptName, trainDfTarget, "trainDfTarget")
inspect_data(currentScriptName, cvDf, "cvDf")
inspect_data(currentScriptName, cvDfTarget, "cvDfTarget")
inspect_data(currentScriptName, testDf, "testDf")

# Сохраняем данные в csv.
save_results(currentScriptName, trainDf, "trainDf")
save_results(currentScriptName, trainDfTarget, "trainDfTarget")
save_results(currentScriptName, cvDf, "cvDf")
save_results(currentScriptName, cvDfTarget, "cvDfTarget")
save_results(currentScriptName, testDf, "testDf")
