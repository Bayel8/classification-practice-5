
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


# ==========================================
# 1. СОЗДАНИЕ ПАПОК И СОХРАНЕНИЕ ДАТАСЕТА
# ==========================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

csv_path = DATA_DIR / "breast_cancer.csv"

if not csv_path.exists():
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.data.copy()

    # 1 = злокачественная опухоль
    # 0 = доброкачественная опухоль
    df["target"] = (dataset.target == 0).astype(int)

    df.to_csv(csv_path, index=False)

    print("Датасет сохранён:", csv_path)


# ==========================================
# 2. ЗАГРУЗКА И АНАЛИЗ ДАННЫХ
# ==========================================

df = pd.read_csv(csv_path)

print("\nПервые 5 строк:")
print(df.head())

print("\nРазмер датасета:", df.shape)

print("\nКоличество классов:")
print(df["target"].value_counts())

print("\nПропущенные значения:", df.isnull().sum().sum())

X = df.drop(columns=["target"])
y = df["target"]

print("\n0 = доброкачественная опухоль")
print("1 = злокачественная опухоль")


# ==========================================
# 3. РАЗДЕЛЕНИЕ ДАННЫХ
# 60% обучение, 20% проверка, 20% тест
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nРазмер обучающей выборки:", X_train.shape)
print("Размер проверочной выборки:", X_val.shape)
print("Размер тестовой выборки:", X_test.shape)


# ==========================================
# 4. ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ
# ==========================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)

print("\nМодель успешно обучена.")


# ==========================================
# 5. ПОЛУЧЕНИЕ ВЕРОЯТНОСТЕЙ
# ==========================================

# predict_proba возвращает вероятности классов 0 и 1
val_probabilities = model.predict_proba(X_val)[:, 1]
test_probabilities = model.predict_proba(X_test)[:, 1]

print("\nПервые 10 вероятностей класса 1:")
print(np.round(test_probabilities[:10], 3))


# ==========================================
# 6. ВЫБОР ПОРОГА С УЧЁТОМ ЦЕНЫ ОШИБКИ
# ==========================================

COST_FN = 5  # Пропустить злокачественную опухоль
COST_FP = 1  # Ложно отметить доброкачественную

thresholds = np.arange(0.05, 0.96, 0.01)

best_threshold = 0.5
best_cost = float("inf")

for threshold in thresholds:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1]
    ).ravel()

    total_cost = COST_FN * fn + COST_FP * fp

    if total_cost < best_cost:
        best_cost = total_cost
        best_threshold = threshold

print("\nЛучший порог:", round(best_threshold, 2))
print("Стоимость ошибок на проверочной выборке:", best_cost)


# ==========================================
# 7. СРАВНЕНИЕ МЕТРИК НА ТЕСТОВОЙ ВЫБОРКЕ
# ==========================================

pred_default = (test_probabilities >= 0.5).astype(int)

pred_tuned = (
    test_probabilities >= best_threshold
).astype(int)


def calculate_metrics(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Error cost": COST_FN * fn + COST_FP * fp
    }


metrics_default = calculate_metrics(y_test, pred_default)
metrics_tuned = calculate_metrics(y_test, pred_tuned)

comparison = pd.DataFrame(
    [metrics_default, metrics_tuned],
    index=["Threshold 0.5", f"Threshold {best_threshold:.2f}"]
)

print("\nСРАВНЕНИЕ МЕТРИК:")
print(comparison.round(4))

comparison.to_csv(RESULTS_DIR / "metrics_comparison.csv")


# ==========================================
# 8. МАТРИЦА ОШИБОК
# ==========================================

cm = confusion_matrix(y_test, pred_tuned, labels=[0, 1])

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign (0)", "Malignant (1)"],
    yticklabels=["Benign (0)", "Malignant (1)"]
)

plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title(f"Матрица ошибок, порог = {best_threshold:.2f}")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()


# ==========================================
# 9. ROC-КРИВАЯ И ROC-AUC
# ==========================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    test_probabilities
)

roc_auc = roc_auc_score(y_test, test_probabilities)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC-кривая")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
plt.close()


# ==========================================
# 10. PR-КРИВАЯ И AVERAGE PRECISION
# ==========================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(y_test, test_probabilities)
)

average_precision = average_precision_score(
    y_test,
    test_probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"Average Precision = {average_precision:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall кривая")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "pr_curve.png", dpi=150)
plt.close()


# ==========================================
# 11. СОХРАНЕНИЕ ИТОГОВ
# ==========================================

summary = [
    "ПРАКТИЧЕСКАЯ РАБОТА № 5",
    "Классификатор и анализ метрик",
    "",
    f"Количество наблюдений: {len(df)}",
    f"Количество признаков: {X.shape[1]}",
    f"ROC-AUC: {roc_auc:.4f}",
    f"Average Precision: {average_precision:.4f}",
    f"Выбранный порог: {best_threshold:.2f}",
    f"Стоимость FN: {COST_FN}",
    f"Стоимость FP: {COST_FP}",
    "",
    "Сравнение метрик:",
    comparison.round(4).to_string(),
    "",
    "ROC-AUC показывает способность модели ранжировать",
    "положительные примеры выше отрицательных.",
    "Чем ближе ROC-AUC к 1, тем лучше разделение классов.",
    "",
    "Average Precision суммирует качество precision-recall",
    "по разным порогам и полезен для оценки положительного класса.",
    "",
    "Порог выбран по стоимости ошибок на проверочной выборке.",
    "Итоговое сравнение проведено на отдельной тестовой выборке."
]

with open(
    RESULTS_DIR / "report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(summary))

print("\nROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(average_precision, 4))
print("\nРезультаты сохранены в папке results/")
print("Практическая работа завершена.")
