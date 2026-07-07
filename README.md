# НИР — Задача 4: Классификация по полу (SVM vs Random Forest)

Учебная научно-исследовательская работа. Вариант 3.

## Описание

Датасет: [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)

Задача: бинарная классификация по признаку **пол (gender)** на основе анкетных данных соискателей.

## Модели

- **SVM** (метод опорных векторов) — подбор гиперпараметров через `GridSearchCV` (kernel, C, gamma, decision_function_shape, shrinking)
- **Random Forest** — итеративный (coarse-to-fine) подбор гиперпараметров: `max_features`, `max_depth`, `criterion`, `min_samples_split`, `min_samples_leaf`, `n_estimators`

Обе модели обучены с `class_weight='balanced'` из-за сильного дисбаланса классов (~90% Male).

## Результаты

| Модель        | Precision | Recall | F1     |
|---------------|-----------|--------|--------|
| SVM           | 0.1554    | 0.5218 | 0.2395 |
| Random Forest | 0.1644    | 0.4847 | 0.2456 |

Лучший классификатор по F1: **Random Forest**.

## Запуск

```bash
pip install pandas numpy scikit-learn
jupyter notebook KMBO-11-24_Bazaev_Ivan__4.ipynb
```
