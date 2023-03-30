# Настраиваем импорты.
import pathlib
import torch
import torch.nn

from sklearn.preprocessing import StandardScaler
from joblib import dump

from eda import *

# Создаём папку для временных файлов.
tmp_folder_path = pathlib.Path("intermediate data/tmp")
tmp_folder_path.mkdir(parents=True, exist_ok=True)

# Получаем имя текущего скрипта для сохранения выводов.
current_script_name = pathlib.Path(__file__).name

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
               "Необходима ли дополнительная информация?", size="large")
print("Применять ансупервайзд не будем.")

show_separator("8. Обучить модель и вывести валидационный скор по метрике качества.", size="large")

# Нормализует столбцы с А, по U.
columns_to_normalize = train_df.columns.values[1:22]
scaler: StandardScaler = StandardScaler()
scaler.fit(train_df[columns_to_normalize])
print(type(scaler))
train_df[columns_to_normalize] = scaler.transform(train_df[columns_to_normalize])
train_df.info()

# Сохраняет скейлер в дамп в папку intermediate data/results/.
filepath = Path(str("intermediate data/results/" + current_script_name + "_scaler_dump"))
filepath.parent.mkdir(parents=True, exist_ok=True)
dump(scaler, filepath, compress=True)

# TODO Собрать модель логистической регрессии.
# Создаём тензоры из датафреймов.
train_tensor: torch.Tensor = torch.Tensor(train_df.drop(['ID'], axis=1).values)
train_target_tensor: torch.Tensor = torch.Tensor(train_target_df.drop(['ID'], axis=1).values)


class LogisticRegression(torch.nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_input_features, 1)

    # sigmoid transformation of the input
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


lr = LogisticRegression(train_tensor.size()[1])

num_epochs = 500
learning_rate = 0.01
# Использует Binary Cross Entropy.
criterion = torch.nn.BCELoss()
# Использует ADAM optimizer.
optimizer = torch.optim.SGD(lr.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    y_pred = lr(train_tensor)
    loss = criterion(y_pred, train_target_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 20 == 0:
        # printing loss values on every 10 epochs to keep track
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# TODO Обучить модель
# TODO Нормализовать тестовые данные.
# TODO Получить результат.
# TODO Проверить результат на CV.
# TODO Проверить результат на тесте (хотя бы на уровне сравнение корреляций).
