import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def prepare_data(data: pd.DataFrame, method="IQR") -> pd.DataFrame:
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Remove rows with null values
    data = data[data.notnull().all(axis=1)]

    # Remove consecutive duplicates
    not_duplicate = data.diff(-1).any(axis=1)
    not_duplicate[not_duplicate.size-1] = True
    data = data[not_duplicate.values]

    # Remove outliers
    if method == "IQR":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif method == "zscore":
        data = data[(np.abs(stats.zscore(data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]
    else:
        raise ValueError("Invalid outlier detection method. Choose 'IQR' or 'zscore'.")

    return data

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default="data.csv", help="path to the csv file with data"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    data = load_data(args.data)
    print(f"Loaded {len(data)} rows of data.")

    data = prepare_data(data, method="IQR")
    print(f"Cleaned data has {len(data)} rows.")

    # Analyze the distribution of GT Turbine decay state coefficient
    gt_turbine_coeff_stats = data['GT Turbine decay state coefficient'].describe()
    print("GT Turbine Decay State Coefficient Statistics:\n", gt_turbine_coeff_stats)

    # Define the threshold for failure ( 0.985 based on data analysis)
    threshold = 0.985

    # Create a binary target based on decay state coefficient
    data['Failure'] = (data['GT Turbine decay state coefficient'] < threshold).astype(int)

    # Check the distribution of the target variable
    print("Target Variable Distribution:\n", data['Failure'].value_counts(normalize=True))

    # Split into features (X) and target (y)
    X = data.drop(columns=['GT Turbine decay state coefficient', 'Failure'])
    y = data['Failure']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Increase max_iter to 1000 to resolve convergence issues
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()

    # Analyze feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    print("Feature Importance:\n", feature_importance)

    # Optional: Save feature importance to a CSV file
    feature_importance.to_csv("feature_importance.csv", index=False)

    # Additional visualization for feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='Coefficient', y='Feature')
    plt.title("Feature Importance")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
