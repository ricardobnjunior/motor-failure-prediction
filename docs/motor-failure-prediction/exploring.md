# Exploring — Motor Failure Prediction

## Context

Technical challenge for a selection process. Goal: build a Python ML notebook predicting motor failure in the next cycle. Evaluation criteria: approach soundness, code clarity, practical applicability — not peak model performance.

## Dataset (`data.csv`)

- **16,138 rows**, **80 motors**
- **25 columns:** `id`, `cycle`, sensors `p00–p20` (21 columns), settings `s1`, `s2`
- Each motor starts at cycle 1 and runs until breakdown
- **Last cycle per motor = failure point**

### Life-span distribution

| Metric  | Value |
|---------|-------|
| Min     | 128 cycles |
| Max     | 362 cycles |
| Mean    | 201.7 cycles |
| Median  | 196 cycles |

### Data quality

- Zero missing values
- All motors begin at cycle 1
- No duplicate rows detected

---

## Sensor Analysis

### Constant sensors — zero predictive value (drop)

| Sensor | Value | Note |
|--------|-------|------|
| p00 | 518.67 | exactly constant |
| p01 | 1.30   | exactly constant |
| p07 | 0.03   | exactly constant |
| p09 | 2388.0 | exactly constant |
| p10 | 100.0  | exactly constant |
| p16 | 14.62  | exactly constant |
| s2  | ≈0.0   | std=0.0003, 13 unique — negligible |

### Variable sensors — correlated with RUL (keep)

Pearson correlation with RUL:

| Sensor | r with RUL | Direction |
|--------|-----------|-----------|
| p02    | -0.705    | rises as motor degrades |
| p15    | -0.687    | rises as motor degrades |
| p03    | +0.679    | drops as motor degrades |
| p05    | -0.311    | mild rise |
| p06    | -0.649    | rises as motor degrades |
| p08    | -0.613    | rises as motor degrades |
| p11    | -0.617    | rises as motor degrades |
| p12    | +0.638    | drops as motor degrades |
| p13    | +0.647    | drops as motor degrades |
| p14    | -0.590    | rises as motor degrades |
| p18    | +0.662    | drops as motor degrades |
| p19    | -0.566    | rises as motor degrades |
| p20    | -0.391    | mild rise |
| s1     | ≈0.000    | noise — drop |

**Active sensor set (13 features):** p02, p03, p04, p05, p06, p08, p11, p12, p13, p14, p15, p18, p20

---

## Degradation Trajectory Confirmed

Clear monotonic degradation visible when bucketing by RUL:

| RUL bucket | p02 mean | p06 mean | p14 mean | p15 mean |
|------------|----------|----------|----------|----------|
| 1–10       | 48.098   | 8.515    | 1601.53  | 1427.25  |
| 11–30      | 47.901   | 8.489    | 1597.40  | 1420.64  |
| 31–60      | 47.688   | 8.461    | 1593.46  | 1413.78  |
| 61+        | 47.425   | 8.427    | 1588.24  | 1405.11  |

Rolling-mean smoothing (window=10) further exposes the trend, reducing noise.

---

## Problem Formulation

- **Type:** Binary classification
- **Target:** `failure_next_cycle = 1` when `RUL == 1` (last cycle before breakdown)
- **Critical challenge:** Severe class imbalance — 80 positives / 16,138 total (0.50%, ratio 1:200)
- **Cost asymmetry:** False negative (missed failure) >> False positive (unnecessary alert)

---

## Open Questions for Research

1. Which engineered features give strongest signal near failure? (rolling stats, linear trend slope, cycle ratio)
2. Best strategy for 1:200 imbalance — SMOTE, class weights, or threshold tuning?
3. How to properly split data respecting motor identity (no motor-leakage across train/test)?
4. Which algorithm handles imbalanced time-series-like data best? (tree-based vs logistic)
5. What decision threshold maximizes recall while keeping precision acceptable?
