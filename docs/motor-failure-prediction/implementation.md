# Implementation — Motor Failure Prediction

## Deliverable

`motor_failure_prediction.ipynb` — fully executed Jupyter notebook, runs top-to-bottom without errors.

## Execution Environment

- Python 3.x
- pandas 2.3.1, numpy 1.26.4, xgboost 1.7.0, scikit-learn 1.3.0
- matplotlib, seaborn, imbalanced-learn (imblearn)
- nbformat, jupyter

## Feature Set (142 total)

14 active sensors × 9 engineered features + cycle_ratio + cycle_log:
- `{sensor}_rm5`, `_rm10` — rolling mean (window 5, 10)
- `{sensor}_rs5`, `_rs10` — rolling std (window 5, 10)
- `{sensor}_rmin10`, `_rmax10` — rolling min/max (window 10)
- `{sensor}_rslope10` — linear regression slope (window 10)
- `{sensor}_ema10` — exponential moving average (span 10)
- `{sensor}_dev` — deviation from motor's early-cycle baseline (first 10 cycles)

## Results

| Model | Threshold | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.934 | 0.812 | 0.342 | 0.481 | 0.9935 |
| Random Forest | 0.077 | 0.812 | 0.163 | 0.271 | 0.9562 |
| XGBoost | 0.010 | **0.938** | 0.155 | 0.265 | 0.9893 |
| XGBoost + SMOTE | 0.307 | 0.812 | **0.191** | **0.310** | 0.9892 |

**Best model for recall:** XGBoost (misses 1 out of 16 failures in holdout)
**Best balanced model:** XGBoost + SMOTE (F1=0.310, better precision)

## Notebook Sections

1. Setup & Imports
2. Data Loading & Overview (3 visualizations)
3. EDA — sensor variability, RUL correlation, degradation trajectory, temporal trends (4 visualizations)
4. Feature Engineering (142 features, leakage-free)
5. Train/Test Split (motor-level, 64/16 motors)
6. Model Training (LogReg, RF, XGB, XGB+SMOTE)
7. Threshold Optimization (PR curve, recall-targeted)
8. Evaluation (confusion matrices, feature importance, model comparison table)
9. Practical Applications
10. Future Improvements

## Figures Generated

- `fig_01_overview.png` — life-span histogram, class balance, boxplot
- `fig_02_correlations.png` — sensor vs RUL Pearson correlations
- `fig_03_degradation.png` — degradation trajectory by RUL bucket
- `fig_04_trajectories.png` — temporal sensor trends (4 sample motors)
- `fig_05_curves.png` — precision-recall + ROC curves (all models)
- `fig_06_confusion_matrices.png` — confusion matrices (4 models)
- `fig_07_feature_importance.png` — XGBoost top 25 features

## Acceptance Criteria Check

- [x] Notebook runs top-to-bottom without errors
- [x] XGBoost recall ≥ 0.75 on test set (achieved 0.938)
- [x] All cells have markdown explanations
- [x] 7 visualizations across all key analyses
- [x] Practical applications section (Section 8)
- [x] Future improvements section (Section 9)
- [x] Motor-level train/test split (no leakage)
- [x] SMOTE comparison included
