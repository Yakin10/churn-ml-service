from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "churn_pipeline.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)

def predict_churn(customer: dict, model=None) -> dict:
    if model is None:
        model = load_model()

    customer_df = pd.DataFrame([customer])

    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0][1])

    return {
        "prediction" : "Churn" if prediction == 1 else "No Churn",
        "churn_probability" : round(probability, 4),
    }

if __name__ == "__main__":
    model = load_model()
    example_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 110.0,
    }

    #result = predict_churn(example_customer)
    #print(result)
    print(predict_churn(example_customer, model))