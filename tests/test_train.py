import pandas as pd

from src.churn_ml.train import build_pipeline

def test_pipeline_fits_and_predicts():
    X = pd.DataFrame({
        "tenure": [1, 12, 24, 36, 48, 60],
        "MonthlyCharges": [25.0, 40.0, 55.0, 65.0, 80.0, 95.0],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two Year",
            "Month-to-month",
            "One year",
            "Two year",
        ],
    })

    y = pd.Series([1, 1, 0, 0, 0, 0])

    model = build_pipeline(X)
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(X), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()