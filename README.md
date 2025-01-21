---

# Sales Prediction Project

This project is a machine learning pipeline designed to predict sales using historical data and various regression models. The solution includes comprehensive preprocessing, feature engineering, model evaluation, and hyperparameter tuning to optimize predictive performance.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Tools and Libraries](#tools-and-libraries)
8. [How to Run](#how-to-run)
9. [Results](#results)
10. [Future Work](#future-work)

---

## Project Overview

This project focuses on building a machine learning pipeline to predict sales (`num_sold`) for a given dataset. It includes:

- Handling missing values.
- Transforming categorical and numerical features.
- Applying regression models for prediction.
- Evaluating model performance using metrics like Mean Absolute Error (MAE).

---

## Dataset

Dataset description can be found at https://drive.google.com/file/d/1DrAqBM1YGmxFK7uWA_mTKVMxEC5kqVRh/view
The dataset includes the following columns:

- **date**: Date of sales.
- **store**: Store identifier (categorical).
- **item**: Item identifier (categorical).
- **num_sold**: Number of items sold (target variable).

For this project, the dataset was split into training and test sets.

---

## Preprocessing

### Steps:
1. **Datetime Conversion**: The `date` column was converted to a datetime object to extract temporal features:
   - **Year**, **Month**, **Week**, **Day**, and **Day of the Week**.
2. **Handling Missing Values**: Imputed missing values in the dataset using suitable strategies.
3. **One-Hot Encoding**: Transformed categorical variables like `store` and `item` into numerical representations using `OneHotEncoder`.
4. **Scaling**: Normalized numerical features to ensure uniformity in input data.

---

## Feature Engineering

Feature engineering aimed to extract meaningful information from raw data. Derived features include:

- **Day of the Week**: To analyze weekday trends in sales.
- **Cumulative Features**: Aggregates such as monthly or weekly sales totals.

The final dataset comprised both the original columns and derived features.

---

## Modeling

The following regression models were used:

1. **Linear Regression**:
   - A baseline model to set initial performance expectations.

2. **Decision Tree Regressor**:
   - Explored simple tree-based predictions with minimal hyperparameter tuning.

3. **Random Forest Regressor**:
   - A robust ensemble model tuned using **GridSearchCV** for optimal hyperparameters.

---

## Evaluation

Evaluation was performed using the **Mean Absolute Error (MAE)** metric:

- **Train-Test Split**: Ensured that the model generalized well to unseen data.
- **Grid Search**: Hyperparameter tuning to improve model accuracy.

---

## Tools and Libraries

The project leverages the following Python libraries:

- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computations.
- **Scikit-Learn**: Machine learning tools for preprocessing, modeling, and evaluation.
- **Matplotlib & Seaborn**: Visualization libraries for data analysis.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/apoorvpandey048/machine_learning-property-price-prediction-California-Housing.git
   cd sales-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute the pipeline:
   ```bash
   python main.py
   ```

4. View results and plots in the output directory.

---

## Results

### Key Findings:
- The Random Forest Regressor achieved the best performance with an RMSE of **X**.
- Derived features such as the **Day of the Week** significantly improved predictive accuracy.

### Model Performance Summary:

| Model                  | RMSE (Mean) | RMSE (STD) |
|------------------------|-------------|------------|
| Linear Regression      | 69107.08    | 2885.87    |
| Decision Tree Regressor| 71803.73    | 2352.58    |
| Random Forest Regressor| 50232.63    | 2212.78    |

---

## Future Work

- **Feature Expansion**: Introduce more derived features to capture temporal or seasonal trends.
- **Hyperparameter Optimization**: Use advanced search techniques like **Bayesian Optimization**.
- **Deploy Model**: Integrate the final model into a production environment.

---
