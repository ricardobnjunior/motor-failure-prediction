# Research — Motor Failure Prediction

## Q1: Feature Engineering Strategy

### Raw features (13 sensors after dropping constants)
p02, p03, p04, p05, p06, p08, p11, p12, p13, p14, p15, p18, p20

### Engineered features per sensor

| Feature Type | Formula | Rationale |
|---|---|---|
| Rolling mean (w=5,10) | `mean(t-w+1 … t)` | Smooths noise, captures trend level |
| Rolling std (w=5,10)  | `std(t-w+1 … t)`  | Captures volatility increase near failure |
| Rolling slope (w=10)  | `linregress(cycles, sensor)` | Explicit degradation rate |
| Cycle ratio | `cycle / 196` (median life) | Normalized age proxy, no leakage |

**Note on leakage:** `max_cycle` per motor is known only in training. In production the model never knows total life span. All features must be computed from past history only (rolling window). Cycle ratio using dataset median (196) is an acceptable prior.

### Top features by expected importance (from EDA correlations)
p02 (r=−0.705), p15 (r=−0.687), p03 (r=+0.679), p18 (r=+0.662), p13 (r=+0.647)

---

## Q2: Class Imbalance Strategy

| Strategy | Pros | Cons | Decision |
|---|---|---|---|
| `class_weight='balanced'` | Simple, no data augmentation, no leakage | Less aggressive | **Primary: use in all models** |
| SMOTE | Synthetic minority samples | Must apply only on train set | **Compare as bonus** |
| Threshold tuning | Fine-grained precision/recall control | Post-hoc | **Always apply after training** |
| Undersampling | Fast | Destroys majority samples | **Discard** |

**Decision:** `class_weight='balanced'` + threshold tuning via precision-recall curve.

---

## Q3: Train/Test Split — Motor Identity Leak Prevention

**Wrong approach:** random row split → motor data leaks across sets, inflated metrics.

**Correct approach:** **Motor-level group split** — entire motor history either in train or test.
- 80 motors → 64 train / 16 test (80/20 by motor ID)
- Cross-validation: `GroupKFold(n_splits=5)` with `groups=motor_id`

---

## Q4: Algorithm Selection

| Algorithm | Handles Imbalance | Feature Interactions | Decision |
|---|---|---|---|
| Logistic Regression | with class_weight | linear only | **Baseline** |
| Random Forest | with class_weight | tree-based | **Benchmark** |
| XGBoost | scale_pos_weight | tree-based, boosted | **Primary model** |

**Primary model: XGBoost** — best documented for predictive maintenance, `scale_pos_weight=200` for 1:200 imbalance.

---

## Q5: Evaluation Metrics and Decision Threshold

Given cost asymmetry (missed failure >> false alarm):

| Metric | Role |
|---|---|
| **Recall** | Primary — must maximize |
| **Precision** | Secondary — false alarms reduce trust |
| **F1** | Harmonic balance |
| **PR-AUC** | Best for imbalanced datasets |
| **ROC-AUC** | Overall discrimination |
| Confusion Matrix | TP/TN/FP/FN breakdown |

**Threshold strategy:** Plot precision-recall curve, select threshold achieving ≥0.80 recall while maximizing precision.

---

## Research Conclusions

| Question | Answer |
|---|---|
| Features | Raw sensors + rolling mean/std/slope (w=5,10) + cycle_ratio |
| Imbalance | class_weight='balanced' + threshold tuning |
| Split | GroupKFold by motor ID — no motor leakage |
| Model | XGBoost primary, Random Forest + LogReg as benchmarks |
| Metric | Recall primary, PR-AUC secondary, threshold optimized on validation |
