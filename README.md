# ML Assignment 2
# Problem Statement

The objective of this assignment is to build and compare multiple machine learning classification models to predict whether an individual earns more than $50K per year based on demographic and employment-related attributes.

The problem is formulated as a binary classification task:

Class 0 → Income ≤ 50K

Class 1 → Income > 50K

The goal is to evaluate and compare six classification models using standard evaluation metrics and identify the best-performing model for this dataset.

# Dataset Description

The dataset used is the Adult Income Dataset (UCI / Kaggle version).

Dataset Characteristics:

Total Features: 14 input features

Target Variable: income

Task Type: Binary Classification

Data Type:

Numerical features (age, capital gain, capital loss, hours per week, etc.)

Categorical features (workclass, education, occupation, marital status, race, sex, etc.)

Target Classes:

<=50K

>50K

The dataset contains socio-economic attributes used to predict annual income category.

Preprocessing steps applied:

Handling missing values (?)

One-hot encoding for categorical variables

Feature scaling where required

Train-test split

Model evaluation using classification metrics

# Models Used and Performance Comparison

The following six models were implemented and evaluated:

Logistic Regression

Decision Tree

k-Nearest Neighbors (kNN)

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)


# Comparision Table:
| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
| ------------------------ | -------- | ----- | --------- | ------ | ----- | ----- |
| Logistic Regression      | 0.765    | 0.919 | 0.884     | 0.61   | 0.722 | 0.557 |
| Decision Tree            | 0.935    | 0.935 | 1.00      | 0.87   | 0.930 | 0.877 |
| kNN                      | 0.765    | 0.927 | 0.884     | 0.61   | 0.722 | 0.557 |
| Naive Bayes              | 0.695    | 0.762 | 0.627     | 0.96   | 0.759 | 0.460 |
| Random Forest (Ensemble) | 0.94     | 0.992 | 1.00      | 0.88   | 0.936 | 0.886 |
| XGBoost (Ensemble)       | 0.815    | 0.951 | 0.932     | 0.68   | 0.786 | 0.654 |
                                                                                                   

## Model Observations

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Provides balanced performance but struggles with recall for the positive class (>50K). Performs moderately well but may underfit complex relationships. |
| Decision Tree | High accuracy and strong recall. Captures nonlinear patterns effectively but may risk overfitting without pruning. |
| kNN | Performance similar to Logistic Regression. Sensitive to feature scaling and high dimensionality due to one-hot encoding. |
| Naive Bayes | High recall but lower precision and accuracy. Tends to over-predict the positive class due to strong independence assumptions. |
| Random Forest (Ensemble) | Best overall performer. Highest AUC (0.992), strong balance across all metrics. Robust and reduces overfitting compared to a single decision tree. |
| XGBoost (Ensemble) | Good AUC and precision. Performs better than linear models but slightly lower accuracy compared to Random Forest. Handles nonlinear relationships efficiently. |


# Final Conclusion

Among all models tested:

Random Forest (Ensemble) achieved the best overall performance.

It produced the highest Accuracy, AUC, F1-score, and MCC.

Ensemble methods outperformed linear and probabilistic models.

Naive Bayes showed strong recall but lower precision.

Logistic Regression and kNN performed similarly but were outperformed by tree-based models.

Therefore, Random Forest is the most suitable model for this dataset based on overall evaluation metrics.
