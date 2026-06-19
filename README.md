# RentNYC — Rental Price Prediction

**Goal:** Predict apartment rental prices in New York based on property characteristics to assist renters and real estate platforms.

---

## Research Questions

1. How do the number of bedrooms and bathrooms affect rental price?
2. Is the interest_level of a listing related to its price?
3. Which model best predicts price — linear regression, decision tree, or a naive baseline?

---

## Data

**Source:** [Kaggle — Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)  
**Size:** ~49,000 rows, 15 columns (3 features + target used for training)

Fields used:

| Field | Type | Description |
|---|---|---|
| `bathrooms` | float | Number of bathrooms |
| `bedrooms` | int | Number of bedrooms |
| `interest_level` | str → int | Interest level: low / medium / high |
| `price` | int | **Target** — monthly rent price ($) |

No missing values. Outliers removed using the 1st and 99th percentiles.

---

## Methodology

**Tech Stack:** Python, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

**Steps:**

* **EDA and Cleaning** — `info()`, `describe()`, `corr()`; outliers detected (max price $4,490,000), removed using 1–99 percentiles.
* **Visualization** — histograms and boxplots for the target; scatterplots of features vs. price; correlation heatmap.
* **Encoding** — `interest_level`: low → 0, medium → 1, high → 2.
* **Feature Engineering** — quadratic features (`bathrooms²`, `bedrooms²`, `interest_level²`) + `PolynomialFeatures(degree=10)`; squares did not improve correlation with the target.
* **Split:** pre-existing `train.json` / `test.json` from the competition organizers.
* **Metrics:** MAE and RMSE on train and test sets.

---

## Models and Results

| Model | MAE train | MAE test | RMSE train | RMSE test |
|---|---|---|---|---|
| `naive_mean` | 1139.193 | 1137.832 | 1597.647 | 1590.391 |
| `naive_median` | 1086.211 | 1084.588 | 1644.236 | 1635.396 |
| `linear_regression` | 784.564 | 790.798 | 1123.878 | 1368.601 |
| **`decision_tree`** | **756.045** | **759.615** | **1077.484** | **1081.242** |

> The decision tree consistently outperforms linear regression and naive baselines on both metrics.

**Key Findings:**

* Correlation between numeric features and price is weak; the strongest is for `bathrooms` and `bedrooms`.
* Latitude and longitude are strongly correlated with each other (r ≈ −0.97), but were not used in training.
* Quadratic features did not improve correlation — the decision tree captures non-linearity better.
* `interest_level`  carries demand information but by itself is a weak predictor of price.

---

## Project Structure

```
.
├── src/RentNYC.ipynb                 # Main notebook
├── src/data/train.json               # Training data (Kaggle)
├── sra/data/test.json                # Test data (Kaggle)
└── README.md
```

---

## How to Reproduce

```bash
# 1. Load data from Kaggle
# https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data
# Put train.json and test.json in src/data

# 2. Run all cells in the notebook
jupyter notebook src/RentNYC.ipynb
```