# Motor Failure Prediction

Binary classification model for predictive maintenance — predicting motor failure in the next operating cycle.

## Problem

Given sensor readings from motors in operation, predict whether a motor will fail in the **next cycle**. The last cycle before breakdown is the ground truth failure label.

## Dataset

- 80 motors · 16,138 observations · 25 columns (id, cycle, 21 sensors, 2 settings)
- Severe class imbalance: 80 failures / 16,138 samples (0.5%, ratio 1:200)

## Approach

1. **EDA** — sensor variability, RUL correlation, degradation trajectory analysis
2. **Feature Engineering** — 142 features: rolling mean/std/slope/min/max, EMA, baseline deviation
3. **Motor-level train/test split** — prevents temporal leakage (64 train / 16 test motors)
4. **Models** — Logistic Regression (baseline), Random Forest, XGBoost, XGBoost + SMOTE
5. **Threshold optimization** — recall-targeted threshold selection (PR curve analysis)

## Results

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.812 | 0.342 | 0.481 | 0.9935 |
| Random Forest | 0.812 | 0.163 | 0.271 | 0.9562 |
| **XGBoost** | **0.938** | 0.155 | 0.265 | **0.9893** |
| XGBoost + SMOTE | 0.812 | 0.191 | 0.310 | 0.9892 |

XGBoost misses **1 out of 16 failures** in the holdout test set.

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyter
jupyter notebook motor_failure_prediction.ipynb
```

## Files

```
motor_failure_prediction.ipynb  ← Main notebook (fully executed)
data.csv                        ← Dataset
docs/motor-failure-prediction/  ← Arche process documentation
```
