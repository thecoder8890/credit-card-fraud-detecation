# How to Run the Credit Card Fraud Detection Script

This document provides instructions on how to run the credit card fraud detection script.

## Prerequisites

1. Python 3.6 or higher
2. Required libraries (install using the requirements.txt file)
3. The credit card dataset

## Installation Steps

1. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Download the dataset**:
   - The script requires a file named `creditcard.csv`
   - You can download it from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place the downloaded CSV file in the same directory as the script

## Running the Script

Once you have installed the dependencies and downloaded the dataset, you can run the script using:

```
python credit_card_fraud_detection.py
```

## Expected Output

The script will:
1. Load and analyze the dataset
2. Preprocess the data
3. Train a Logistic Regression model
4. Train an XGBoost model
5. Evaluate both models and display performance metrics

The output will include:
- Dataset information
- Class distribution
- Model evaluation metrics (Accuracy, Sensitivity, Specificity, F1-Score, ROC-AUC)
- Classification reports
- ROC curve plots saved as image files ('logistic_regression_roc.png' for Logistic Regression and 'xgboost_roc_curve.png' for XGBoost)

## Troubleshooting

If you encounter any issues:

1. **Missing dataset error**:
   - Make sure the `creditcard.csv` file is in the same directory as the script
   - Check that the file name is exactly "creditcard.csv" (case sensitive)

2. **Library import errors**:
   - Ensure you've installed all dependencies using the requirements.txt file
   - Try installing any missing libraries individually using pip

3. **Memory errors**:
   - The dataset is large and may require significant memory
   - Close other applications to free up memory
   - If running on a system with limited RAM, consider using a smaller subset of the data
