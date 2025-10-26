# Predicting Bike Sharing Demand: A Comparative Study of Ridge and Lasso Regression for Regularization and Feature Selection

## 📖 Abstract

This project offers an **in-depth exploration of regularized linear regression techniques**, specifically focusing on **Ridge ($\text{L}_2$)** and **Lasso ($\text{L}_1$)** regularization. Utilizing a comprehensive bike sharing dataset, the study demonstrates how these methods effectively address **multicollinearity** and prevent **overfitting** in high-dimensional feature spaces.

A core emphasis is placed on the comparative analysis of:
* **Lasso Regression's feature selection capability** by driving irrelevant feature coefficients to absolute zero.
* **Ridge Regression's coefficient shrinkage** that uniformly reduces the magnitude of all coefficients.

This comparison highlights their distinct behaviors and practical applications in robust predictive modeling.

---

## 🎯 Objectives

The primary goals of this comparative study are:

* To implement and compare **Ordinary Least Squares (OLS)**, **Ridge**, and **Lasso** regression models for predicting hourly bike rental counts.
* To analyze how different **regularization strengths ($\alpha$ values)** affect model coefficients and prediction performance (Bias-Variance trade-off).
* To demonstrate Lasso's **feature selection capability** by identifying the most important predictors through coefficient shrinkage to zero.
* To visualize **regularization paths** for both Ridge and Lasso regression across a wide range of $\alpha$ values.
* To evaluate the trade-offs between **bias and variance** introduced by the different regularization techniques.

---

## 💾 Dataset

| Property | Details |
| :--- | :--- |
| **Dataset Name** | Bike Sharing Dataset |
| **Source** | UCI Machine Learning Repository - Open Source |
| **URL** | [https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) |
| **Description** | Contains 17,379 hourly records of bike rental data spanning two years, including weather, seasonal, and temporal features influencing demand. |
| **Target Variable** | **`cnt`** - Total count of bike rentals (casual + registered users). |

### Features

| Category | Features |
| :--- | :--- |
| **Temporal** | `season`, `year`, `month`, `hour`, `holiday`, `weekday`, `workingday` |
| **Weather** | `temp`, `feelslike_temp`, `humidity`, `windspeed` |
| **Derived** | `weathersit` (Weather situation: clear, mist, rain/snow) |

---

## ⚙️ Methodology

The project follows a rigorous four-step methodology:

### 1. Data Preprocessing & Feature Engineering
* **Handle Datetime Features:** Extract and transform cyclical features (e.g., hour, month) using **sine/cosine transformations** to preserve continuity.
* **Categorical Encoding:** Appropriately encode categorical variables (e.g., `weathersit`).
* **Multicollinearity Check:** Examine features using **Variance Inflation Factor (VIF)**.
* **Feature Complexity:** Create **polynomial features (degree=2)** to intentionally increase model complexity, making the need for regularization evident.
* **Data Splitting:** Split data into training (70%), validation (15%), and test (15%) sets.

### 2. Model Development
* **Baseline Model:** Ordinary Least Squares (OLS) regression using all engineered features.
* **Regularized Models:**
    * **Ridge Regression ($\text{L}_2$ penalty)**
    * **Lasso Regression ($\text{L}_1$ penalty)**
    * **ElasticNet ($\text{L}_1$ and $\text{L}_2$ combined)** for an advanced comparison.

### 3. Hyperparameter Tuning & Analysis
* **Optimal $\alpha$ Selection:** Use **GridSearchCV** with cross-validation to find the optimal $\alpha$ (regularization strength) for each model.
* **Visualization:** Plot **regularization paths** to visualize how coefficients change as $\alpha$ increases.
* **Feature Selection Analysis:** Analyze the count of features selected (non-zero coefficients) by Lasso at various $\alpha$ thresholds.

### 4. Model Evaluation & Interpretation
* **Performance Metrics:** Evaluate final models on the held-out test set using **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R-squared ($R^2$)**.
* **Coefficient Comparison:** Compare the final coefficient magnitudes and patterns among OLS, Ridge, and Lasso.
* **Feature Importance:** Identify the most influential predictors based on Lasso's selection.
* **Residual Analysis:** Analyze residual patterns to assess if regularization has improved model assumptions and fit quality.

---

## 🚀 Expected Outcomes & Deliverables

* **Implementation** of OLS, Ridge, Lasso, and ElasticNet regression models.
* **Visualization** of regularization paths and detailed coefficient behaviors.
* **Analysis of feature importance** based on Lasso's selection mechanism.
* A **Performance Comparison Table** highlighting the trade-offs in RMSE, MAE, and $R^2$ between different regularization approaches.
* **Final Report** discussing practical guidelines and scenarios for choosing between $\text{L}_1$ (Lasso) and $\text{L}_2$ (Ridge) regularization.

---

## 🛠️ Tools and Libraries

| Category | Tools/Libraries |
| :--- | :--- |
| **Programming Language** | Python |
| **Core Libraries** | Scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Statistical Analysis** | Statsmodels |

---

## 🗓️ Project Timeline

| Week | Focus Area |
| :--- | :--- |
| **Week 1** | Data loading, **Exploratory Data Analysis (EDA)**, and advanced feature engineering (cyclical encoding, polynomial features). |
| **Week 2** | Baseline OLS implementation and initial Ridge/Lasso model development. |
| **Week 3** | **Hyperparameter tuning** (GridSearchCV), regularization path analysis, and ElasticNet implementation. |
| **Week 4** | Comprehensive model comparison, final **feature importance analysis**, and report generation. |