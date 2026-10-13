from fastapi.testclient import TestClient

from src.churn_ml.api import app

client = TestClient(app)

def test_home_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()

def test_prediction_endpoint():
    customer = {
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

    with TestClient(app) as client:
        response = client.post("/predict", json=customer)

        assert response.status_code == 200, response.text

        result = response.json()
        assert result["prediction"] in {"Churn", "No Churn"}
        assert 0 <= result["churn_probability"] <=1

def test_invalid_contract_is_rejected():
    customer = {
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
        "Contract": "InvalidValue",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 110.0,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=customer)

        assert response.status_code == 422