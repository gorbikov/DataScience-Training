# Настраиваем импорты.
import pandas as pd

# Функция для визуального разделителя.
def printseparator():
    print('----------------------------------------------------------')

# Вводные.
trainPath = 'data/exam_module_4/1_variant_dna_sequence_mutation_prediction/input/train.csv'
testPath = 'data/exam_module_4/1_variant_dna_sequence_mutation_prediction/input/test.csv'
cvFraction = 0.15
randomCeed = 777


# 1. Определяем тип задачи.
print('Task type: logistic regression or SVM')
printseparator()

# 2. Создаём фреймы и выделяем часть датасета на CV.
trainDf = pd.read_csv(trainPath).sample(frac=(1 - cvFraction), random_state=randomCeed).drop('mutation', axis=1)
trainDfTarget = pd.read_csv(trainPath).sample(frac=(1 - cvFraction), random_state=randomCeed)[['ID', 'mutation']]
cvDf = pd.read_csv(trainPath).drop(trainDf.index).drop('mutation', axis=1)
cvDfTarget = pd.read_csv(trainPath).drop(trainDfTarget.index)[['ID', 'mutation']]
testDf = pd.read_csv(testPath)

print('Original train data: ' + str(pd.read_csv(trainPath).shape))
print('Original test data: ' + str(pd.read_csv(testPath).shape))
print('Train data: ' + str(trainDf.shape))
print('Train target data: ' + str(trainDfTarget.shape))
print('CV data: ' + str(cvDf.shape))
print('CV target data: ' + str(cvDfTarget.shape))
print('Test data: ' + str(testDf.shape))

printseparator()

# 3. Определяем тип переменных в датасете.
print('Train data types: \n' + str(trainDf.dtypes))
print('Train target data types: \n' + str(trainDfTarget.dtypes))
printseparator()

# 4. Если это необходимо провести препроцессинг данных, нужно ли применять алгоритмы понижения размерности?
# Нужно ли убирать аномалии?
print('Так как n << m, лучше использовать логистическую регрессию, либо SMV without kernel.')

# Переводим первые 9 столбцов в цифры (по методу one-hot, one-hot столбцы добавляются в конце датафрейма).
dummieCounter = 0
for col in trainDf.columns:
    if trainDf[col].dtypes == object:
        dummieCounter += len(trainDf[col].unique())
        print('Unique in ' + str(col) + ': ' + str(len(trainDf[col].unique())))
print('Dummie columns: ' + str(dummieCounter))
printseparator()

trainDf = pd.get_dummies(trainDf)
cvDf = pd.get_dummies(cvDf)
testDf = pd.get_dummies(testDf)

# 5. Провести EDA и вывести какие-то умозаключения и посмотреть на распределения признаков, на корреляции, на выбросы.

# 6. Подумать над вариантом модели для того чтобы решить задачу (либо ансамблем моделей)

# 7. Подумать нужно ли применять Unsupervised learning подход для решения задачи? Неоходима ли дополнительная информация?

# 8. Обучить модель и вывести валидационный скор по метрике качества.

# 9. Построить отчет на 10 предложнений.

# 10. Выйти и объяснить подход к решению задачи.








































































































