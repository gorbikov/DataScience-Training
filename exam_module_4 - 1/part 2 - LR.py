# Настраиваем импорты.
import pathlib
import pandas as pd
from os import path

import torch
import torch.nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from joblib import dump

from eda import *

# Создаём папку для временных файлов.
tmp_folder_path = pathlib.Path("intermediate data/tmp")
tmp_folder_path.mkdir(parents=True, exist_ok=True)

# Создаём папку для результатов.
results_folder_path = pathlib.Path("intermediate data/results")
results_folder_path.mkdir(parents=True, exist_ok=True)

# Получаем имя текущего скрипта для сохранения выводов.
current_script_name = pathlib.Path(__file__).name

# Вводные.
cv_path = pathlib.Path("intermediate data/results/part 1.py_cv_df.csv")
cv_target_path = pathlib.Path("intermediate data/results/part 1.py_cv_df_target.csv")
test_path = pathlib.Path("intermediate data/results/part 1.py_test_df.csv")
train_path = pathlib.Path("intermediate data/results/part 1.py_train_df.csv")
train_target_path = pathlib.Path("intermediate data/results/part 1.py_train_df_target.csv")

# Создаёт датафреймы.
cv_df = pd.read_csv(cv_path, index_col='ID')
cv_target_df = pd.read_csv(cv_target_path, index_col='ID')
test_df = pd.read_csv(test_path, index_col='ID')
train_df = pd.read_csv(train_path, index_col='ID')
train_target_df = pd.read_csv(train_target_path, index_col='ID')











# Добавляем новые фичи на основе имеющихся.
for column1 in train_df.loc[:, "A":"U"].columns:
    for column2 in train_df.loc[:, "A":"U"].columns:
        new_column = {column1 + "*" + column2: train_df[column1] * train_df[column2]}
        new_column = pd.DataFrame(new_column)
        train_df = pd.concat([train_df, new_column], axis=1)
        print(column1 + "*" + column2 + " Train DONE!")

        new_column = {column1 + "*" + column2: cv_df[column1] * cv_df[column2]}
        new_column = pd.DataFrame(new_column)
        cv_df = pd.concat([cv_df, new_column], axis=1)
        print(column1 + "*" + column2 + " CV DONE!")


train_df.head().to_csv(tmp_folder_path.joinpath("train.csv"))
cv_df.head().to_csv(tmp_folder_path.joinpath("cv.csv"))










show_separator("6. Подумать над вариантом модели, для того чтобы решить задачу (либо ансамблем моделей)",
               size="large")
print('Так как n << m, лучше использовать логистическую регрессию, либо SMV without kernel.'
      'Я буду использовать логистическую регрессию.')

show_separator("7. Подумать нужно ли применять Unsupervised learning подход для решения задачи? "
               "Необходима ли дополнительная информация?", size="large")
print("Применять ансупервайзд не будем.")

show_separator("8. Обучить модель и вывести валидационный скор по метрике качества.", size="large")

# Нормализует столбцы с А, по U.
columns_to_normalize = train_df.columns.values[1:22]
scaler: StandardScaler = StandardScaler()
scaler.fit(train_df[columns_to_normalize])
train_df[columns_to_normalize] = scaler.transform(train_df[columns_to_normalize])
show_separator("Столбцы train_df нормализованы.")

# Сохраняет скейлер в дамп в папку intermediate data/results/.
filepath = Path(str("intermediate data/results/" + current_script_name + "_scaler_dump"))
filepath.parent.mkdir(parents=True, exist_ok=True)
dump(scaler, filepath, compress=True)
show_separator("Скейлер сохранен в файл в папке results.")

# Создаёт тензоры из датафреймов.
cv_tensor: torch.Tensor = torch.Tensor(cv_df.drop(['ID'], axis=1).values)
cv_target_tensor: torch.Tensor = torch.Tensor(cv_target_df.drop(['ID'], axis=1).values)
test_tensor: torch.Tensor = torch.Tensor(test_df.drop(['ID'], axis=1).values)
train_tensor: torch.Tensor = torch.Tensor(train_df.drop(['ID'], axis=1).values)
train_target_tensor: torch.Tensor = torch.Tensor(train_target_df.drop(['ID'], axis=1).values)


# Собирает модель логистической регрессии.
class LogisticRegression(torch.nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_input_features, 1)

    # sigmoid transformation of the input
    def forward(self, x):
        y_prediction = torch.sigmoid(self.linear(x))
        return y_prediction


lr = LogisticRegression(train_tensor.size()[1])

# Задаёт параметры обучения.
num_epochs = 1000
learning_rate = 0.001
# Использует Binary Cross Entropy.
criterion = torch.nn.BCELoss()
# Использует ADAM optimizer.
optimizer = torch.optim.SGD(lr.parameters(), lr=learning_rate)

# Загружает модель из файла.
if path.exists(results_folder_path.joinpath(current_script_name + '_model_weights')):
    lr.load_state_dict(torch.load(results_folder_path.joinpath(current_script_name + '_model_weights')))
    show_separator("Параметры модели загружены из файла.")
else:
    show_separator("Файл с параметрами подели отсутствует. Обучение начинается с нуля.")

# Начинает обучение.
show_separator("Обучение модели на " + str(num_epochs) + " эпохах:")
loss_function_values_for_graph = dict()
previous_loss_function_value = None
for epoch in range(num_epochs):
    y_pred = lr(train_tensor)
    loss = criterion(y_pred, train_target_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 100 == 0:
        # Выводит loss function каждый 20 эпох.
        loss_function_values_for_graph[epoch + 1] = loss.item()
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
    if (previous_loss_function_value is not None) and (float(loss.item()) > previous_loss_function_value):
        show_separator("!!!Обучение остановлено, т.к. зафиксирован рост lost function.!!!")
        break
    previous_loss_function_value = float(loss.item())

# Сохраняет в файл график loss function.
generate_loss_function_graph(current_script_name, loss_function_values_for_graph)

# Сохраняет параметры модели в файл.
torch.save(lr.state_dict(), results_folder_path.joinpath(current_script_name + '_model_weights'))
show_separator("Параметры модели сохранены в папке results.")

# Выводит метрики результата для train.
show_separator("Текущие метрики для train:")
with torch.no_grad():
    target_predicted = lr(train_tensor)
    target_predicted_class = target_predicted.round()
    acc = target_predicted_class.eq(train_target_tensor).sum() / float(train_target_tensor.shape[0])
    print(f'accuracy: {acc.item():.4f}')
    print(classification_report(train_target_tensor, target_predicted_class))
    confusion_matrix_train = confusion_matrix(train_target_tensor, target_predicted_class)
    print(confusion_matrix_train)

# Выводит метрики результата для CV.
show_separator("Текущие метрики для CV:")
with torch.no_grad():
    target_predicted = lr(cv_tensor)
    target_predicted_class = target_predicted.round()
    acc = target_predicted_class.eq(cv_target_tensor).sum() / float(cv_target_tensor.shape[0])
    print(f'accuracy: {acc.item():.4f}')
    print(classification_report(cv_target_tensor, target_predicted_class))
    confusion_matrix_cv = confusion_matrix(cv_target_tensor, target_predicted_class)
    print(confusion_matrix_cv)

# TODO Добиться точности на CV на уровне 95%+.
# TODO Посмотреть какие метрики могут позволить выявить проблему.

# TODO Нормализует тестовые данные.
# TODO Сохраняет матрицу корреляций для предсказанных данных.
