# Hotel Booking Cancellation Prediction

## 1. Project Overview
This project builds a machine learning pipeline to predict whether a hotel booking will be canceled. The solution includes data cleaning, feature engineering, class imbalance handling, model training with XGBoost, evaluation, and CI/CD automation using GitHub Actions.

---

## 2. Dataset Description
The dataset contains hotel reservation records including:
- Booking details
- Stay duration
- Guest composition
- Pricing information
- Market segment and customer history

The target variable is:
- `booking_status` (0 = Not Canceled, 1 = Canceled)

Dataset Source: Kaggle – Hotel Reservations Classification Dataset

---
---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Data Summary
- Dataset shape, data types, missing values, and duplicate records were analyzed.
- Numerical feature distributions, target distribution, and correlation heatmaps were generated.
- All EDA visualizations are saved in `artifacts/eda/`.

### 3.2 Potential Data Issues Identified for Cleaning
Exploratory Data Analysis revealed several real-world data quality issues. Duplicate booking records were present and were removed to avoid data leakage and biased learning. Missing values were observed in both numerical and categorical attributes, which could distort model training if left unhandled; numerical values were imputed using the median to reduce the impact of extreme values, while categorical values were filled using the mode. Certain numerical attributes such as lead time and average room price exhibited heavy right-skew and extreme values, indicating the presence of outliers; these were treated using percentile-based capping (winsorization). Data type inconsistencies were also found, with categorical attributes stored as object types and numerical values stored as mixed types; these were explicitly converted to appropriate numeric, categorical, and datetime formats to ensure schema consistency.

---

## 4. Data Cleaning
- Duplicate records removed
- Missing values imputed (median for numeric, mode for categorical)
- Outliers capped using 1st and 99th percentiles
- Explicit numeric, categorical, and datetime type conversion applied

---

## 5. Feature Engineering

### 5.1 Engineered Features
- Total stay nights
- Total guests
- Lead time category (short / medium / long)
- Average price per person
- Weekend stay indicator
- Previous cancellation ratio
- Family indicator

### 5.2 How Features Improve Model Performance
Feature engineering was performed to enhance the predictive power of the models by capturing hidden behavioral and economic patterns in hotel bookings. Total stay nights represent overall trip commitment, where longer stays generally reduce cancellation likelihood. Total guests capture group size, where larger groups cancel less frequently due to higher coordination cost. Lead time was discretized into short, medium, and long categories to capture its non-linear impact on cancellations. Average price per person normalizes room cost by group size to better model price sensitivity. The weekend stay indicator differentiates leisure travel, which is more prone to cancellation than business travel. Previous cancellation ratio captures historical customer reliability, and the family indicator reflects higher planning rigidity. These features reduce noise, improve interpretability, and significantly enhance classification performance.

---

## 6. Outlier Detection and Treatment
Outliers were detected in numerical features such as:
- Lead time
- Average room price

These were treated using:
- Percentile-based capping (winsorization at 1st and 99th percentiles)

This prevents extreme values from skewing model learning.

---

## 7. Encoding of Categorical Variables
- Nominal categorical features were encoded using One-Hot Encoding
- Ordinal categories were encoded using ordered mapping
- Encoding was applied within a preprocessing pipeline to avoid data leakage

---

## 8. Class Imbalance Handling
The target variable was imbalanced. This was handled using:
- SMOTE for Random Forest and XGBoost
- Class weights for Logistic Regression
- `scale_pos_weight` parameter for XGBoost

---

## 9. Models Trained
The following models were trained and compared:

- Logistic Regression (baseline)
- Random Forest
- XGBoost (final production model)

Hyperparameter tuning was applied to XGBoost using RandomizedSearchCV.

---

## 10. Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

All evaluation artifacts are stored in:
artifacts/eval/

This includes:
- Classification report
- Confusion matrix plot
- ROC curve
- Feature importance plot

---

## 11. Most Important Features & Business Interpretation
Feature importance analysis from the final tuned XGBoost model showed that lead time, average price per person, previous cancellation ratio, market segment type, and total stay nights were the most influential predictors of cancellation. Lead time indicates that very early and last-minute bookings are at higher risk of cancellation. Average price per person reflects price sensitivity. Previous cancellation ratio highlights repeat cancellers. Market segment differentiates cancellation behavior across booking channels. Total stay nights captures commitment strength. From a business perspective, these insights support strategic overbooking, targeted confirmations, dynamic pricing, and customer reliability scoring to reduce revenue loss.

---

## 12. Model Saving & Usage
The final trained model is saved as: models/best_model.joblib

---

## 13. CI/CD Automation

A GitHub Actions CI/CD workflow was implemented to automate:
-> Dependency installation from requirements.txt
-> End-to-end pipeline execution
-> Model validation
-> Model saving
-> Automatic artifact upload of trained model

---
---
