import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import shap


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_data(data: pd.DataFrame, method="IQR") -> pd.DataFrame:
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Sanitize column names for compatibility with XGBoost
    data.columns = data.columns.str.replace(r'[\[\]<>]', '', regex=True)

    # Remove rows with null values
    data = data[data.notnull().all(axis=1)]

    # Remove consecutive duplicates
    not_duplicate = data.diff(-1).any(axis=1)
    not_duplicate[not_duplicate.size - 1] = True
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


def plot_roc_curve(fpr, tpr, model_name, roc_auc):
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')


def main():
    args = parse_arguments()
    data = load_data(args.data)
    print(f"Loaded {len(data)} rows of data.")

    data = prepare_data(data, method="IQR")
    print(f"Cleaned data has {len(data)} rows.")

    # Analyze the distribution of GT Turbine decay state coefficient
    gt_turbine_coeff_stats = data['GT Turbine decay state coefficient'].describe()
    print("GT Turbine Decay State Coefficient Statistics:\n", gt_turbine_coeff_stats)

    # Define the threshold for failure (updated to 0.985 based on data analysis)
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

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    plt.figure(figsize=(10, 6))

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{model_name} Cross-Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Evaluate the model
        print(f"\n{model_name} Classification Report:\n", classification_report(y_test, y_pred))
        print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, model_name, roc_auc)

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            print(f"\n{model_name} Feature Importance:\n", feature_importance)

    # Hyperparameter tuning for Random Forest
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    print(f"Best Random Forest Parameters: {rf_grid.best_params_}")

    # Hyperparameter tuning for XGBoost
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_params,
                            cv=3, scoring='accuracy')
    xgb_grid.fit(X_train, y_train)
    print(f"Best XGBoost Parameters: {xgb_grid.best_params_}")

    # Interpretability with SHAP for XGBoost
    explainer = shap.Explainer(xgb_grid.best_estimator_)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curves for Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
