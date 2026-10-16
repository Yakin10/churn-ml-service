from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.churn_ml.predict import load_model, predict_churn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    model = None

app = FastAPI(
    title = "Customer Churn Prediction API",
    description = "Predict Customer churn using a trained ML Pipeline.",
    version = "1.0.0",
    lifespan=lifespan,
)


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
    
    @field_validator(
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    )
    @classmethod
    def validate_yes_no_fields(cls, value):
        if value not in {"Yes", "No"}:
            raise ValueError("Value must be Yes or No")
        return value

    @field_validator(
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    )
    @classmethod
    def validate_yes_no_or_no_phone_service(cls, value):
        if value not in {"Yes", "No", "No phone service", "No internet service"}:
            raise ValueError("Invalid service option")
        return value

    @field_validator("InternetService")
    @classmethod
    def validate_internet_service(cls, value):
        if value not in {"DSL", "Fiber optic", "No"}:
            raise ValueError("Invalid InternetService")
        return value

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment_method(cls, value):
        allowed = {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        }
        if value not in allowed:
            raise ValueError("Invalid PaymentMethod")
        return value


@app.on_event("startup")
def load_model_on_startup():
    global model
    model = load_model()

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: Customer, threshold: float = 0.5):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )
    
    if not 0 <= threshold <= 1:
        raise HTTPException(
            status_code=422,
            detail="Threshold must be between 0 and 1."
        )
    
    try:
        result = predict_churn(customer.model_dump(), model, threshold=threshold)
        logger.info("Prediction completed successfully.")
        return result
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to internal server error."
        )