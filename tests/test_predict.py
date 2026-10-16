from src.churn_ml.predict import predict_churn
import pytest

class FixedProbabilityModel:
    """A test model that always predicts a 40% churn probability"""

    def predict_proba(self, customer_df):
        return [[0.60, 0.40]]

def test_prediction_returns_expected_fields(trained_test_model):
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

    result = predict_churn(customer, trained_test_model)

    assert result['prediction'] in {"Churn", "No Churn"}
    assert 0.0 <= result["churn_probability"] <= 1.0

def test_lower_threshold_predicts_churn():
    result = predict_churn(
        customer = {},
        model = FixedProbabilityModel(),
        threshold = 0.3
    )

    assert result["prediction"] == "Churn"
    assert result["churn_probability"] == 0.4

def test_default_threshold_remains_05():
    result = predict_churn(
        customer = {},
        model = FixedProbabilityModel()
    )

    assert result["prediction"] == "No Churn"

def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        predict_churn(
            customer = {},
            model = FixedProbabilityModel,
            threshold = 1.5
        )