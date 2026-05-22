# Planning — Motor Failure Prediction

## Notebook Structure

Single Jupyter Notebook: `motor_failure_prediction.ipynb`

### Section 0 — Setup & Imports
- Import all libraries (pandas, numpy, sklearn, xgboost, matplotlib, seaborn, imblearn)
- Set random seeds for reproducibility

### Section 1 — Data Loading & Initial Exploration
- Load `data.csv`
- Shape, dtypes, missing value check
- Per-motor life-span distribution (histogram)
- Class balance overview

### Section 2 — Exploratory Data Analysis
- Sensor variability table (std, unique values)
- Correlation of sensors with RUL (bar chart)
- Degradation trajectory plots: 4–6 sensors across RUL buckets
- Per-motor sensor trend visualization (sample motors)

### Section 3 — Feature Engineering
- Drop constant sensors: p00, p01, p07, p09, p10, p16, s2
- Compute `RUL = max_cycle - cycle` per motor
- Compute `failure_next_cycle` target (RUL == 1)
- Sort by (id, cycle) before rolling features
- Rolling mean and std (window=5, 10) per sensor per motor
- Rolling linear slope (window=10) per sensor per motor
- `cycle_ratio = cycle / 196`
- Final feature set overview

### Section 4 — Train/Test Split
- Motor-level split (no leakage): first 64 motors train, last 16 test
- Drop RUL from features (leakage)
- Class balance after split

### Section 5 — Model Training
- Baseline: Logistic Regression (class_weight='balanced')
- Benchmark: Random Forest (class_weight='balanced')
- Primary: XGBoost (scale_pos_weight=200)
- Each model: fit + predict_proba on test set

### Section 6 — Threshold Optimization
- Plot precision-recall curve for XGBoost
- Find optimal threshold (maximize F1 or target recall ≥ 0.80)
- Apply threshold to generate final predictions

### Section 7 — Evaluation & Comparison
- Confusion matrix (heatmap) for all 3 models
- Classification report (precision, recall, F1)
- ROC-AUC and PR-AUC comparison table
- Feature importance plot (XGBoost top-20)

### Section 8 — Practical Applications
- Narrative on real-world deployment scenarios
- Maintenance scheduling, alert systems, cost reduction
- Limitations and deployment considerations

### Section 9 — Future Improvements
- LSTM/GRU for temporal patterns
- Multi-horizon prediction (fail in next N cycles)
- Survival analysis (Weibull)
- Anomaly detection as complementary signal
- Online learning for continuous calibration

---

## File Layout

```
/
├── data.csv
├── motor_failure_prediction.ipynb
├── task.txt
└── docs/motor-failure-prediction/
    ├── exploring.md
    ├── research.md
    ├── planning.md
    └── implementation.md
```

## Acceptance Criteria

- Notebook runs top-to-bottom without errors
- XGBoost recall ≥ 0.75 on test set
- All cells have markdown explanations
- Visualizations: degradation trends, confusion matrix, feature importance, PR curve
- Practical applications section present
- Future improvements section present
