from typing import Any, Literal
from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    employee_id_ref: str | None = Field(default=None)
    Age: int = Field(default=20, ge=18, le=70)
    JobRole: str = Field(default="Sales Representative")
    MaritalStatus: str = Field(default="Single")
    MonthlyIncome: int = Field(default=3000, gt=0)
    YearsAtCompany: int = Field(default=1, ge=0)
    EnvironmentSatisfaction: int = Field(default=2, ge=1, le=4)
    OverTime: Literal["Yes", "No"] = Field(default="Yes")


class PredictionResponse(BaseModel):
    record_id: int
    api_key_masked: str
    attrition_prediction: str
    resign_probability: float
    saved_to_db: bool


class PredictionLogItem(BaseModel):
    id: int
    created_at: str
    #employee_id_ref: str | None
    api_key_masked: str
    attrition_prediction: str
    resign_probability: float
    input_features: dict[str, Any]