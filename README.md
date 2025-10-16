# Heart Failure Ensemble Analysis 

This project predicts **heart disease** using **Decision Trees**, **Random Forest**, and **XGBoost**.  
I integrated **GridSearchCV** to find the best combination of key hyperparameters and visualized model performance with clear **train vs. validation accuracy plots**.
Dataset: Kaggle – Heart Failure Prediction Dataset: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
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
 
Decision Tree — Val accuracy: 0.8571
RF GridSearchCV — best params: {'max_depth': 12, 'min_samples_split': 2, 'n_estimators': 200}
XGB GridSearchCV — best params: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}

Selected model (Val): XGBoost (best GS) — Val: 0.8750 | Test: 0.8696

Results Summary

Both ensemble methods performed better than the single Decision Tree.
XGBoost slightly outperformed Random Forest, showing more stable validation accuracy across parameter sweeps.




