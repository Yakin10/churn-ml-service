import pandas as pd
import pytest

from src.churn_ml.train import build_pipeline


@pytest.fixture
def trained_test_model():
    base_customer = {
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

    customers = []

    for tenure, monthly, contract in [
        (2, 55.0, "Month-to-month"),
        (12, 65.0, "One year"),
        (24, 75.0, "Two year"),
        (3, 60.0, "Month-to-month"),
        (48, 80.0, "Two year"),
        (6, 50.0, "One year"),
    ]:
        customer = base_customer.copy()
        customer["tenure"] = tenure
        customer["MonthlyCharges"] = monthly
        customer["TotalCharges"] = tenure * monthly
        customer["Contract"] = contract
        customers.append(customer)

    X = pd.DataFrame(customers)
    y = [1, 0, 0, 1, 0, 1]

    model = build_pipeline(X)
    model.fit(X, y)

    return model