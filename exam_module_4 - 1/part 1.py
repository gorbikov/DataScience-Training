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

separator_show('1. Определяем тип задачи.', type="large")
print('Task type: logistic regression or SVM')

separator_show("2. Создаём фреймы и выделяем часть датасета на CV.", type="large")
# Формируем оригинальные датафреймы
originalTrainCvDf = pd.read_csv(trainPath, index_col='ID')
originalTestDf = pd.read_csv(testPath, index_col='ID')

separator_show("3. Определить тип переменных в датасете.", type="large")
# Смотрим огригинальные датафреймы.
data_inspection(originalTrainCvDf, currentScriptName, "originalTrainCvDf")
duplicates_search(originalTrainCvDf, "originalTrainCvDf")
duplicates_delete(originalTrainCvDf, "originalTrainCvDf")

data_inspection(originalTestDf, currentScriptName, "originalTestDf")
duplicates_search(originalTestDf, "originalTestDf")
duplicates_delete(originalTestDf, "originalTestDf")

separator_show("""4. Если это необходимо провести препроцессинг данных, нужно ли применять алгоритмы понижения
размерности? Нужно ли убирать аномалии?""", type="large")

# Переводим первые 9 столбцов в цифры (по методу one-hot, one-hot столбцы добавляются в конце датафрейма).
unique_counter_for_object_type(originalTrainCvDf, "originalTrainCVDf")
unique_counter_for_object_type(originalTestDf, "originalTestDf")
trainCvDf = pd.get_dummies(originalTrainCvDf)
testDf = pd.get_dummies(originalTestDf)

separator_show("""5. Провести EDA и вывести какие-то умозаключения и посмотреть на распределения признаков, на
корреляции, на выбросы.""", type='large')

correlation_with_target(trainCvDf, "trainCvDf", "mutation")

# Формируем датафреймы из псевдорандомных выборок.
trainDf = trainCvDf.sample(frac=(1 - cvFraction), random_state=randomCeed).drop('mutation', axis=1)
trainDfTarget = trainCvDf.sample(frac=(1 - cvFraction), random_state=randomCeed)[['mutation']]
cvDf = trainCvDf.drop(trainDf.index).drop('mutation', axis=1)
cvDfTarget = trainCvDf.drop(trainDfTarget.index)[['mutation']]

# Просмотр данных перед сохранением.
data_inspection(trainDf, currentScriptName, "trainDf")
data_inspection(trainDfTarget, currentScriptName, "trainDfTarget")
data_inspection(cvDf, currentScriptName, "cvDf")
data_inspection(cvDfTarget, currentScriptName, "cvDfTarget")
data_inspection(testDf, currentScriptName, "testDf")

# Сохраняем данные в csv.
results_save(trainDf, currentScriptName, "trainDf")
results_save(trainDfTarget, currentScriptName, "trainDfTarget")
results_save(cvDf, currentScriptName, "cvDf")
results_save(cvDfTarget, currentScriptName, "cvDfTarget")
results_save(testDf, currentScriptName, "testDf")
