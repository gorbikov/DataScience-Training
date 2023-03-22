# Настраиваем импорты.
import pathlib
from eda import *

# Вводные.
trainPath = pathlib.Path('data/exam_module_4/1_variant_dna_sequence_mutation_prediction/input/train.csv')
testPath = pathlib.Path('data/exam_module_4/1_variant_dna_sequence_mutation_prediction/input/test.csv')
cvFraction = 0.15
# Фиксируем рандом.
randomCeed = 777
# Получаем имя текущего скрипта для сохранения выводов.
currentScriptName = pathlib.Path(__file__).name

separator_show('1. Определяем тип задачи.', type="large")
print('Task type: logistic regression or SVM')

separator_show("2. Создаём фреймы и выделяем часть датасета на CV.", type="large")
# Формируем оригинальные датафреймы
originalTrainDf = pd.read_csv(trainPath, index_col='ID')
originalTestDf = pd.read_csv(testPath, index_col='ID')

separator_show("3. Определить тип переменных в датасете.", type="large")
# Смотрим огригинальные датафреймы.
data_inspection(originalTrainDf, currentScriptName, "originalTrainDf")
duplicates_search(originalTrainDf, "originalTrainDf")
duplicates_delete(originalTrainDf, "originalTrainDf")

data_inspection(originalTestDf, currentScriptName, "originalTestDf")
duplicates_search(originalTestDf, "originalTestDf")
duplicates_delete(originalTestDf, "originalTestDf")

# Формируем датафреймы из псевдорандомных выборок.
trainDf = originalTrainDf.sample(frac=(1 - cvFraction), random_state=randomCeed).drop('mutation', axis=1)
trainDfTarget = originalTrainDf.sample(frac=(1 - cvFraction), random_state=randomCeed)[['mutation']]
cvDf = originalTrainDf.drop(trainDf.index).drop('mutation', axis=1)
cvDfTarget = originalTrainDf.drop(trainDfTarget.index)[['mutation']]
testDf = originalTestDf

separator_show("""4. Если это необходимо провести препроцессинг данных, нужно ли применять алгоритмы понижения размерности?
Нужно ли убирать аномалии?""", type="large")

# Переводим первые 9 столбцов в цифры (по методу one-hot, one-hot столбцы добавляются в конце датафрейма).

unique_counter_for_object_type(trainDf, "trainDf")
trainDf = pd.get_dummies(trainDf)
cvDf = pd.get_dummies(cvDf)
testDf = pd.get_dummies(testDf)

data_inspection(trainDf, currentScriptName, "trainDf")
data_inspection(cvDf, currentScriptName, "cvDf")
data_inspection(testDf, currentScriptName, "testDf")

# Сохраняем данные в csv.
results_save(trainDf, currentScriptName, "trainDf")
results_save(cvDf, currentScriptName, "cvDf")
results_save(testDf, currentScriptName, "testDf")
