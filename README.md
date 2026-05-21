# Fraud Detection Analysis with Gradient Boosting

## Project Overview

This project focuses on building and evaluating machine learning models for fraud detection using a banking transaction dataset. The goal is to identify fraudulent transactions with high accuracy, precision, and recall, ultimately leading to a robust system for preventing financial losses.

## Dataset

The analysis is based on a simulated banking transaction dataset, which includes various features such as `transaction_amount`, `device_risk_score`, `account_age_days`, `payment_channel`, and a `fraud_flag` indicating fraudulent transactions. The dataset was obtained from Kaggle.

## Analysis Workflow

1.  **Data Exploration (EDA):** Initial analysis to understand data distributions, identify correlations, and visualize relationships between features and the `fraud_flag`.
2.  **Data Preprocessing:** Handled categorical features through one-hot encoding and split the data into training and testing sets while maintaining class balance.
3.  **Model Training & Evaluation:**
    *   **Random Forest Classifier:** A baseline model was trained and evaluated.
    *   **Gradient Boosting Classifier:** Another powerful ensemble model was trained and evaluated.
4.  **Hyperparameter Tuning:** The Gradient Boosting model was further optimized using `GridSearchCV` to find the best performing parameters.

## Key Findings & Model Performance

Both Random Forest and Gradient Boosting models demonstrated strong performance. The **Tuned Gradient Boosting Classifier** showed the best overall performance:

*   **Best Parameters:** `learning_rate=0.01`, `max_depth=3`, `n_estimators=300`, `subsample=0.8`
*   **ROC AUC Score (Test Set):** 0.9781
*   **Accuracy:** 0.9515
*   **Precision:** 0.8201
*   **Recall:** 0.7840
*   **F1-Score:** 0.8016

`anomaly_score` consistently emerged as the most important feature across all models, highlighting its critical role in identifying fraudulent activities.

## How to Use the Saved Model

To load and use the `tuned_gradient_boosting_model.pkl` for making predictions on new data, you can use the following Python code snippet:

```python
import pickle
import pandas as pd

# Assuming you have new data for prediction (e.g., a DataFrame named 'new_data_df')
# Ensure 'new_data_df' has the same columns as the training data, excluding 'fraud_flag'

# Load the saved model
with open('tuned_gradient_boosting_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions
predictions = loaded_model.predict(new_data_df)
probabilities = loaded_model.predict_proba(new_data_df)[:, 1]

# You can then interpret the predictions and probabilities
for i, pred in enumerate(predictions):
    print(f"Prediction for sample {i+1}: {'Fraud' if pred else 'No Fraud'}, Probability of Fraud: {probabilities[i]:.4f}")
