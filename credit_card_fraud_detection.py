# Credit Card Fraud Detection
# This script is converted from the Jupyter notebook credit_card_fraud_detection.ipynb

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from xgboost import XGBClassifier
# Set matplotlib to use a non-interactive backend
plt.switch_backend('agg')
import warnings
warnings.filterwarnings('ignore')

# Define ROC Curve function
def draw_roc(actual, probs, filename='roc_curve.png', title='Receiver Operating Characteristic'):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()
    print(f"ROC curve saved as '{filename}'")
    return None

# Set pandas display options
pd.set_option('display.max_columns', 500)

def main():
    print("Credit Card Fraud Detection Analysis")
    print("====================================")

    # Check if the dataset exists
    if not os.path.exists('creditcard.csv'):
        print("Error: The dataset file 'creditcard.csv' is missing.")
        print("Please download the dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("and place it in the same directory as this script.")
        sys.exit(1)

    # Reading the dataset
    print("\nLoading dataset...")
    df = pd.read_csv('creditcard.csv')
    print("Dataset loaded successfully!")

    # Display basic information
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset shape:", df.shape)

    print("\nChecking for missing values:")
    df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
    print(df_missing_columns)

    # Class distribution
    print("\nClass distribution:")
    classes = df['Class'].value_counts()
    print(classes)

    normal_share = round((classes[0]/df['Class'].count()*100),2)
    fraud_share = round((classes[1]/df['Class'].count()*100),2)

    print(f"Normal transactions: {normal_share}%")
    print(f"Fraudulent transactions: {fraud_share}%")

    # Creating separate dataframes for fraud and non-fraud
    data_fraud = df[df['Class'] == 1]
    data_non_fraud = df[df['Class'] == 0]

    # Dropping the Time column
    print("\nDropping the Time column as it doesn't show specific patterns for fraud detection")
    df.drop('Time', axis=1, inplace=True)

    # Putting feature variables into X
    X = df.drop(['Class'], axis=1)

    # Putting target variable to y
    y = df['Class']

    print("\nSplitting data into train and test sets (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)

    # Feature Scaling for Amount column
    scaler = StandardScaler()
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount'] = scaler.transform(X_test[['Amount']])

    print("\nApplied StandardScaler to the Amount column")

    # Mitigate skewness with PowerTransformer
    print("\nApplying PowerTransformer to normalize the data distribution...")
    cols = X_train.columns
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)
    X_train[cols] = pt.fit_transform(X_train)
    X_test[cols] = pt.transform(X_test)

    # Model building - Logistic Regression
    print("\nBuilding Logistic Regression model...")

    # Using a simple model with default parameters for demonstration
    logistic_model = LogisticRegression(C=0.01)
    logistic_model.fit(X_train, y_train)

    # Predictions on test set
    y_test_pred = logistic_model.predict(X_test)

    # Evaluation metrics
    confusion = metrics.confusion_matrix(y_test, y_test_pred)
    TP = confusion[1,1]  # true positive 
    TN = confusion[0,0]  # true negatives
    FP = confusion[0,1]  # false positives
    FN = confusion[1,0]  # false negatives

    print("\nLogistic Regression Model Evaluation:")
    print("Confusion Matrix:")
    print(confusion)
    print("\nAccuracy:", metrics.accuracy_score(y_test, y_test_pred))
    print("Sensitivity:", TP / float(TP+FN))
    print("Specificity:", TN / float(TN+FP))
    print("F1-Score:", f1_score(y_test, y_test_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # ROC Curve
    y_test_pred_proba = logistic_model.predict_proba(X_test)[:,1]
    auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
    print("ROC-AUC Score:", auc)

    # Save ROC curve plot
    print("Generating ROC curve for Logistic Regression model...")
    draw_roc(y_test, y_test_pred_proba, 'logistic_regression_roc.png', 'Logistic Regression ROC Curve')

    # XGBoost model
    print("\nBuilding XGBoost model...")

    # Using a simple model with default parameters for demonstration
    params = {
        'learning_rate': 0.2,
        'max_depth': 2, 
        'n_estimators': 200,
        'subsample': 0.9,
        'objective': 'binary:logistic'
    }

    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)

    # Predictions on test set
    y_test_pred = xgb_model.predict(X_test)

    # Evaluation metrics
    confusion = metrics.confusion_matrix(y_test, y_test_pred)
    TP = confusion[1,1]  # true positive 
    TN = confusion[0,0]  # true negatives
    FP = confusion[0,1]  # false positives
    FN = confusion[1,0]  # false negatives

    print("\nXGBoost Model Evaluation:")
    print("Confusion Matrix:")
    print(confusion)
    print("\nAccuracy:", metrics.accuracy_score(y_test, y_test_pred))
    print("Sensitivity:", TP / float(TP+FN))
    print("Specificity:", TN / float(TN+FP))
    print("F1-Score:", f1_score(y_test, y_test_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # ROC Curve
    y_test_pred_proba = xgb_model.predict_proba(X_test)[:,1]
    auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
    print("ROC-AUC Score:", auc)

    # Save ROC curve plot with a custom filename
    print("Generating ROC curve for XGBoost model...")
    draw_roc(y_test, y_test_pred_proba, 'xgboost_roc_curve.png', 'XGBoost ROC Curve')

    print("\nAnalysis complete! The XGBoost model generally performs better for credit card fraud detection.")
    print("For a more comprehensive analysis, please refer to the Jupyter notebook.")

if __name__ == "__main__":
    main()
