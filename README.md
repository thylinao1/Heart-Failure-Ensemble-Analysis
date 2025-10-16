# Heart Failure Ensemble Analysis 

This project predicts **heart disease** using **Decision Trees**, **Random Forest**, and **XGBoost**.  
I integrated **GridSearchCV** to find the best combination of key hyperparameters and visualized model performance with clear **train vs. validation accuracy plots**.

---

## 🧩 Project Overview

I used the **Heart Failure Prediction dataset** from Kaggle (`heart.csv`). Workflow:
1. **Load and preprocess** the dataset (convert categorical to numeric if needed).  
2. **Split** the data into Train / Validation / Test subsets (with stratification).  
3. Train a **Decision Tree baseline**.  
4. Explore **Random Forest** and **XGBoost** performance by:
   - Plotting **Train vs Validation accuracy** while sweeping key parameters.  
   - Using **GridSearchCV** to tune:
     - For Random Forest: `n_estimators`, `max_depth`, `min_samples_split`  
     - For XGBoost: `n_estimators`, `max_depth`, `learning_rate`
5. **Compare results** on validation and test sets, and print confusion matrices and reports.

---

## 🧠 Models Used

| Model | Technique | Tuned Parameters |
|--------|------------|------------------|
| Decision Tree | Baseline | — |
| Random Forest | Ensemble Bagging | `n_estimators`, `max_depth`, `min_samples_split` |
| XGBoost | Ensemble Boosting | `n_estimators`, `max_depth`, `learning_rate` |

---

## 📊 Visualizations
I plotted **Train vs Validation accuracy** curves for each parameter:
- Random Forest:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
- XGBoost:
  - `n_estimators`
  - `max_depth`
  - `learning_rate`

These visualizations helped confirm where the models start to overfit and how complexity impacts accuracy.

---


