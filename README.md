# 🏠 House Price Prediction Project

This project builds a machine learning pipeline to predict house prices using various regression models. It includes data preprocessing, feature engineering, model training, and evaluation using cross-validation.

---

## 📌 Project Overview

The goal of this project is to:
- Analyze housing data
- Handle missing values
- Perform feature encoding and scaling
- Train multiple regression models
- Compare model performance using R² score
- Select the best performing model

## ⚙️ Technologies Used

- Python 🐍
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost

## 📊 Workflow

### 1. Data Loading
- Load train and test datasets
- Combine for uniform preprocessing

### 2. Exploratory Data Analysis (EDA)
- Missing value heatmap
- Feature statistics
- Distribution analysis

### 3. Data Preprocessing
- Missing value imputation:
  - Numerical → Median
  - Categorical → Mode
- One-hot encoding for categorical variables
- Feature scaling using StandardScaler

### 4. Model Training
Models used:
- Linear Regression
- Support Vector Regressor (SVR)
- SGD Regressor
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- MLP Regressor
- Gaussian Process Regressor

### 5. Model Evaluation
- 7-Fold Cross Validation
- Evaluation metric: R² Score
### 6. Model Selection
- Best model selected based on highest average R² score

## 📈 Results

After training multiple models, the best performing model is automatically selected based on cross-validation R² score.

