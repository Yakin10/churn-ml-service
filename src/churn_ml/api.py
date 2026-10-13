from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.churn_ml.predict import load_model, predict_churn

app = FastAPI(
    title = "Customer Churn Prediction API",
    description = "Predict Customer churn using a trained ML Pipeline.",
    version = "1.0.0",
)

model = None


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value):
        if value not in {"Male", "Female"}:
            raise ValueError("gender must be Male or Female")
        return value
    
    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, value):
        allowed = {"Month-to-month","One year", "Two year"}
        if value not in allowed:
            raise ValueError(f"Contract must be one of {sorted(allowed)}")
        return value
    
    @field_validator("tenure")
    @classmethod
    def validate_tenure(cls, value):
        if value > 100:
            raise ValueError("tenure must be between 0 and 100 months")
        return value


@app.on_event("startup")
def load_model_on_startup():
    global model
    model = load_model()

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )
    #try:
    #    return predict_churn(customer.model_dump())
    #except FileNotFoundError as exc:
    #    raise HTTPException(status_code=503, detail=str(exc)) from exc
    return predict_churn(customer.model_dump(), model)