import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Завантаження даних
data = pd.read_csv('bioresponse.csv')
X = data.drop(columns=['Activity'])  # ознаки
y = data['Activity']  # цільовий стовпець

# Розділення на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Дрібне дерево рішень
shallow_tree = DecisionTreeClassifier(max_depth=3)
shallow_tree.fit(X_train, y_train)

# Глибоке дерево рішень
deep_tree = DecisionTreeClassifier(max_depth=20)
deep_tree.fit(X_train, y_train)

# Випадковий ліс на дрібних деревах
shallow_forest = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42)
shallow_forest.fit(X_train, y_train)

# Випадковий ліс на глибоких деревах
deep_forest = RandomForestClassifier(max_depth=20, n_estimators=100, random_state=42)
deep_forest.fit(X_train, y_train)

# Функція для розрахунку метрик з урахуванням налаштування порогу
def calculate_metrics(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Log Loss': log_loss(y_test, model.predict_proba(X_test))
    }
    return metrics

# Розрахунок метрик для кожної моделі
metrics_shallow_tree = calculate_metrics(shallow_tree, X_test, y_test)
metrics_deep_tree = calculate_metrics(deep_tree, X_test, y_test)
metrics_shallow_forest = calculate_metrics(shallow_forest, X_test, y_test)
metrics_deep_forest = calculate_metrics(deep_forest, X_test, y_test)

# Виведення результатів
print("Shallow Decision Tree", metrics_shallow_tree)
print("Deep Decision Tree", metrics_deep_tree)
print("Shallow Random Forest", metrics_shallow_forest)
print("Deep Random Forest ", metrics_deep_forest)

# Функція для побудови графіків Precision-Recall та ROC-кривих
def plot_curves(models, model_names, X_test, y_test):
    plt.figure(figsize=(12, 10))

    # Precision-Recall Curve
    plt.subplot(2, 1, 1)
    for model, name in zip(models, model_names):
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=f"{name} (Precision-Recall)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    # ROC Curve
    plt.subplot(2, 1, 2)
    for model, name in zip(models, model_names):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (ROC), AUC = {auc(fpr, tpr):.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Список моделей та їх імен
models = [shallow_tree, deep_tree, shallow_forest, deep_forest]
model_names = ["Shallow Decision Tree", "Deep Decision Tree", "Shallow Random Forest", "Deep Random Forest"]

# Побудуємо графіки для всіх моделей
plot_curves(models, model_names, X_test, y_test)
plt.close()  # Закрыть фигуру после отображения

# Налаштування порогу для мінімізації помилок II роду
best_threshold = 0.3
metrics_deep_forest = calculate_metrics(deep_forest, X_test, y_test, threshold=best_threshold)
print("Метрики для Deep Random Forest при порозі:", best_threshold, metrics_deep_forest)

