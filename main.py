# Імпорт необхідних бібліотек
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt

# 1. Відкрити та зчитати наданий файл з даними
data = pd.read_csv('WQ-R.csv', sep=';')

# 2. Визначити та вивести кількість записів та кількість полів у наборі даних
num_records, num_fields = data.shape
num_records, num_fields

# 3. Вивести перші 10 записів набору даних
data.head(10)

# 4. Розділити набір даних на навчальну та тестову вибірки
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 5. Побудувати модель дерева прийняття рішень
clf_g = DecisionTreeClassifier(max_depth=5, criterion='gini')
clf_g.fit(X_train, y_train)

clf_e = DecisionTreeClassifier(max_depth=5, criterion='entropy')
clf_e.fit(X_train, y_train)

# 6. Представити графічно побудоване дерево
dot_data = export_graphviz(clf_g, out_file=None, feature_names=X.columns, class_names=True, filled=True)
graph = graphviz.Source(dot_data)
graph.render('decision_tree')
graph

dot_data = export_graphviz(clf_e, out_file=None, feature_names=X.columns, class_names=True, filled=True)
graph = graphviz.Source(dot_data)
graph.render('decision_tree')
graph

# 7. Обчислити класифікаційні метрики та представити результати графічно

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# Функція для обчислення метрик
def compute_metrics(y_true, y_pred):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mcc': [],
        'balanced_accuracy': [],
        'youden_j': []
    }

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    youden_j = recall + (1 - (1 - precision)) - 1  # Youden's J = Sensitivity + Specificity - 1

    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['mcc'].append(mcc)
    metrics['balanced_accuracy'].append(balanced_acc)
    metrics['youden_j'].append(youden_j)

    return pd.DataFrame(metrics)

# Прогнозування значень
y_train_pred = clf_g.predict(X_train)
y_test_pred = clf_g.predict(X_test)

# Обчислення метрик для тренувальної та тестової вибірок
metrics_train = compute_metrics(y_train, y_train_pred)
metrics_test = compute_metrics(y_test, y_test_pred)

# Вивід метрик для тренувальної та тестової вибірок
print("Train Metrics")
print(metrics_train)
print("\nTest Metrics")
print(metrics_test)

# Графічне відображення метрик
metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'balanced_accuracy', 'youden_j']
train_metrics_values = [metrics_train[metric][0] for metric in metrics_names]
test_metrics_values = [metrics_test[metric][0] for metric in metrics_names]

x = range(len(metrics_names))

plt.figure(figsize=(12, 6))
plt.bar(x, train_metrics_values, width=0.4, label='Train', align='center')
plt.bar(x, test_metrics_values, width=0.4, label='Test', align='edge')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics for Gini')
plt.xticks(x, metrics_names, rotation=45)
plt.legend()
plt.show()


# Прогнозування значень
y_train_pred = clf_e.predict(X_train)
y_test_pred = clf_e.predict(X_test)

# Обчислення метрик для тренувальної та тестової вибірок
metrics_train = compute_metrics(y_train, y_train_pred)
metrics_test = compute_metrics(y_test, y_test_pred)

# Вивід метрик для тренувальної та тестової вибірок
print("Train Metrics")
print(metrics_train)
print("\nTest Metrics")
print(metrics_test)

# Графічне відображення метрик
metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'balanced_accuracy', 'youden_j']
train_metrics_values = [metrics_train[metric][0] for metric in metrics_names]
test_metrics_values = [metrics_test[metric][0] for metric in metrics_names]

x = range(len(metrics_names))

plt.figure(figsize=(12, 6))
plt.bar(x, train_metrics_values, width=0.4, label='Train', align='center')
plt.bar(x, test_metrics_values, width=0.4, label='Test', align='edge')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics for Entropy')
plt.xticks(x, metrics_names, rotation=45)
plt.legend()
plt.show()

# Прогнозування значень
y_train_pred = clf_g.predict(X_test)
y_test_pred = clf_e.predict(X_test)

# Обчислення метрик для тренувальної та тестової вибірок
metrics_train = compute_metrics(y_test, y_train_pred)
metrics_test = compute_metrics(y_test, y_test_pred)

# Вивід метрик для тренувальної та тестової вибірок
print("Train Metrics")
print(metrics_train)
print("\nTest Metrics")
print(metrics_test)

# Графічне відображення метрик
metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'balanced_accuracy', 'youden_j']
train_metrics_values = [metrics_train[metric][0] for metric in metrics_names]
test_metrics_values = [metrics_test[metric][0] for metric in metrics_names]

x = range(len(metrics_names))

plt.figure(figsize=(12, 6))
plt.bar(x, train_metrics_values, width=0.4, label='Gini', align='center')
plt.bar(x, test_metrics_values, width=0.4, label='Entropy', align='edge')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics')
plt.xticks(x, metrics_names, rotation=45)
plt.legend()
plt.show()

# 8. З’ясувати вплив глибини дерева та мінімальної кількості елементів в листі
depths = range(1, 21)
train_accuracies = []
test_accuracies = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    train_accuracies.append(metrics.balanced_accuracy_score(y_train, clf.predict(X_train)))
    test_accuracies.append(metrics.balanced_accuracy_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(depths, train_accuracies, label='Train BACC')
plt.plot(depths, test_accuracies, label='Test BACC')
plt.xlabel('Depth of Tree')
plt.ylabel('BACC')
plt.legend()
plt.show()

min_elems = range(1, 100)
train_accuracies = []
test_accuracies = []

for min_elem in min_elems:
    clf = DecisionTreeClassifier(min_samples_leaf=min_elem)
    clf.fit(X_train, y_train)
    train_accuracies.append(metrics.balanced_accuracy_score(y_train, clf.predict(X_train)))
    test_accuracies.append(metrics.balanced_accuracy_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(min_elems, train_accuracies, label='Train BACC')
plt.plot(min_elems, test_accuracies, label='Test BACC')
plt.xlabel('Min samples leaf')
plt.ylabel('BACC')
plt.legend()
plt.show()

# 9. Стовпчикова діаграма важливості атрибутів
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title('Важливість атрибутів')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
