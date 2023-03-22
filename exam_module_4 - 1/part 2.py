# Настраиваем импорты.
import pathlib
from eda import *

# Получаем имя текущего скрипта для сохранения выводов.
currentScriptName = pathlib.Path(__file__).name

# Вводные.
trainPath = pathlib.Path()
testPath = pathlib.Path()
cvPath =pathlib.Path("results/part 1.py_cvDf.csv")


# 6. Подумать над вариантом модели, для того чтобы решить задачу (либо ансамблем моделей)

print('Так как n << m, лучше использовать логистическую регрессию, либо SMV without kernel.')

# 7. Подумать нужно ли применять Unsupervised learning подход для решения задачи?
# Неоходима ли дополнительная информация?

# 8. Обучить модель и вывести валидационный скор по метрике качества.
