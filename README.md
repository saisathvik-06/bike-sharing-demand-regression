# Bike Sharing Demand Prediction (Regression from Scratch)

This project predicts hourly bike sharing demand using linear and polynomial regression models implemented from scratch (without using scikit-learn).

It focuses on capturing non-linear patterns and interaction effects in real-world demand data.

---

## Project Overview

- Dataset: Kaggle Bike Sharing Demand  
- Samples: 10,886 hourly observations  
- Task: Supervised regression (predict bike demand count)  
- Best Model: Quadratic Regression with Interaction Terms  
- Best Performance:  
  - R² = 0.566  
  - MSE = 14,338.41  

This project demonstrates:
- End-to-end machine learning pipeline design
- Mathematical understanding of regression
- Proper data leakage prevention
- Bias–variance trade-off analysis

---

## Methodology

### Feature Engineering
- Extracted temporal features from datetime:
  - hour, month, year, weekday
- One-hot encoded categorical variables:
  - season, weather
- Final feature count:
  - 16 base features
  - 152 features after quadratic interactions

### Leakage Prevention
- Removed `casual` and `registered` columns  
  (`count = casual + registered`, which would leak target information)

### Train–Test Split
- Custom random split (no sklearn)
- 80% train / 20% test
- Fixed random seed for reproducibility

---

## Models Implemented

- Linear Regression (Normal Equation)
- Polynomial Regression (degree 2, 3, 4)
- Quadratic Regression with Interaction Terms

All models are implemented from scratch using NumPy.

---

## Results

| Model | MSE | R² |
|------|-----|----|
| Linear Regression | 19,794.64 | 0.401 |
| Polynomial (d=2) | 16,387.01 | 0.504 |
| Polynomial (d=3) | 14,529.97 | 0.560 |
| Polynomial (d=4) | 14,474.13 | 0.562 |
| Quadratic + Interactions | 14,338.41 | 0.566 |

This represents a 41% improvement in R² over linear regression.

---

## Analysis

- Polynomial models reduce bias compared to linear regression
- Interaction terms capture relationships such as:
  - Temperature × Hour
  - Weather × Working Day
- Residual plots show no systematic patterns, indicating a good model fit

---

## Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  

No machine learning libraries (e.g., scikit-learn) were used for modeling.

---

## Execution

```bash
python main_q1.py


