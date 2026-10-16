from fastapi.testclient import TestClient

import src.churn_ml.api as api

def test_home_endpoint(trained_test_model, monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: trained_test_model
    )
    with TestClient(api.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()

def test_prediction_endpoint(trained_test_model, monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: trained_test_model
    )

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

    with TestClient(api.app) as client:
        response = client.post("/predict", json=customer)

        assert response.status_code == 200, response.text

        result = response.json()
        assert result["prediction"] in {"Churn", "No Churn"}
        assert 0 <= result["churn_probability"] <=1

def test_invalid_contract_is_rejected(trained_test_model, monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: trained_test_model
    )

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

    with TestClient(api.app) as client:
        response = client.post("/predict", json=customer)

        assert response.status_code == 422

def test_negative_tenure_is_rejected(trained_test_model, monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: trained_test_model
    )

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -2,
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

    with TestClient(api.app) as client:
        response = client.post("/predict", json=customer)
   
    assert response.status_code == 422

def test_missing_required_field_is_rejected(trained_test_model, monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: trained_test_model
    )

    customer = {
        "gender": "Female",
        #"SeniorCitizen": 0, Missing Senior cetizen field
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

    with TestClient(api.app) as client:
        response = client.post("/predict", json=customer)
  
    assert response.status_code == 422

def test_zero_tenure_is_accepted(trained_test_model, monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: trained_test_model
    )

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 0, #tenure set to zero
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
        "TotalCharges": 0.0,
    }

    with TestClient(api.app) as client:
        response = client.post("/predict", json=customer)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction"] in {"Churn", "No Churn"}
    assert 0 <= result["churn_probability"] <= 1

def test_partner_value_is_rejected(trained_test_model, monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: trained_test_model
    )

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Many",
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

    with TestClient(api.app) as client:
        response = client.post("/predict", json=customer)

    assert response.status_code == 422

def test_invalid_payment_method_is_rejected(trained_test_model, monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: trained_test_model
    )

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
        "PaymentMethod": "RuPay",
        "MonthlyCharges": 55.0,
        "TotalCharges": 110.0,
    }

    with TestClient(api.app) as client:
        response = client.post("/predict", json=customer)

    assert response.status_code == 422

def test_prediction_returns_503_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        api, "load_model", lambda: None
    )

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

    with TestClient(api.app) as client:
        response = client.post("/predict", json=customer)
    
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded."