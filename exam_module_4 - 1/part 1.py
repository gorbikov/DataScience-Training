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

# 4. Если это необходимо провести препроцессинг данных, нужно ли применять алгоритмы понижения размерности?
# Нужно ли убирать аномалии?
separator_show("""4. Если это необходимо провести препроцессинг данных, нужно ли применять алгоритмы понижения размерности?
Нужно ли убирать аномалии?""", type="large")

# Переводим первые 9 столбцов в цифры (по методу one-hot, one-hot столбцы добавляются в конце датафрейма).
dummieCounter = 0
for col in trainDf.columns:
    if trainDf[col].dtypes == object:
        dummieCounter += len(trainDf[col].unique())
        print('Unique in ' + str(col) + ': ' + str(len(trainDf[col].unique())))
print('Dummie columns: ' + str(dummieCounter))

trainDf = pd.get_dummies(trainDf)
cvDf = pd.get_dummies(cvDf)
testDf = pd.get_dummies(testDf)

# 5. Провести EDA и вывести какие-то умозаключения и посмотреть на распределения признаков, на корреляции, на выбросы.

# 6. Подумать над вариантом модели, для того чтобы решить задачу (либо ансамблем моделей)

print('Так как n << m, лучше использовать логистическую регрессию, либо SMV without kernel.')

# 7. Подумать нужно ли применять Unsupervised learning подход для решения задачи?
# Неоходима ли дополнительная информация?

# 8. Обучить модель и вывести валидационный скор по метрике качества.
