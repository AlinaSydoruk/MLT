import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from LogisticRegressionWithCustomLoss import LogisticRegressionWithCustomLoss
# Завантажуємо дані з текстового файлу
data = pd.read_csv('data_banknote_authentication.txt', header=None)


# Припускаємо, що останній стовпець це мітки класів
X = data.iloc[:, :-1].values  # Всі стовпці, окрім останнього
y = data.iloc[:, -1].values   # Останній стовпець

#Нормалізуємо ознаки для покращення збіжності градієнтного спуску.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Розділяємо дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення і тренування моделей з різними функціями втрат
loss_functions = ['logistic', 'adaboost', 'binary_crossentropy']
models = {}

for loss in loss_functions:
    model = LogisticRegressionWithCustomLoss(loss_function=loss, learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train, X_test, y_test)
    models[loss] = model
    model.plot_learning_curves()  # Візуалізація кривих навчання
    model.evaluate(X_test, y_test)  # Оцінка точності на тестових даних
