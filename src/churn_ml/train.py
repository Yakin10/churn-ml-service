from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_Path = ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_Path)

    #Convert numeric-looking charges and handle missing values
    df["TotalCharges"] = pd.to_numeric(df['TotalCharges'], errors="coerce")

    return df

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Identifying Column Type from Feature Data
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

def main() -> None:
    df = load_data()

    x = df.drop(columns=["Churn", "customerID"])
    y = df['Churn'].map({"No" : 0, "Yes" : 1})

    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn Column")

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train,Y_train)

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Pipeline trained successfully.")

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:,1]

    print("\n--- Baseline Evaluation ---")
    print(f"Accuracy : {accuracy_score(Y_test, predictions):.3f}")
    print(f"Precision : {precision_score(Y_test, predictions, zero_division=0):.3f}")
    print(f"Recall : {recall_score(Y_test, predictions, zero_division=0):.3f}")
    print(f"F1 score : {f1_score(Y_test, predictions, zero_division=0):.3f}")
    print(f"ROC_AUC : {roc_auc_score(Y_test, predictions):.3f}")

    print("\n--- confusion Matrix ---")
    print("Rows = actual [No Churn, Churn]")
    print("Column = predicted [No Churn, Churn]")
    print(confusion_matrix(Y_test, predictions, labels=[0,1]))

    print("\n--- Classification Report ---")
    print(classification_report(
        Y_test, predictions, labels=[0,1], target_names=["No Churn", "Churn"], 
        zero_division=0
    ))

if __name__ == "__main__":
    main()