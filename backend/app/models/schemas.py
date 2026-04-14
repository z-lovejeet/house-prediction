from pydantic import BaseModel, Field
from typing import Optional

class HouseInput(BaseModel):
    area: float = Field(..., gt=0, description="Total sq.ft", examples=[1500])
    bedrooms: int = Field(..., ge=1, le=20, examples=[3])
    bathrooms: int = Field(..., ge=1, le=20, examples=[2])
    location: str = Field(..., min_length=1, examples=["Whitefield"])
    balcony: float = Field(default=1.0, ge=0, examples=[2])
    model: Optional[str] = Field(
        default=None,
        description="Model to use: linear, ridge, lasso, elasticnet. Default = best.",
        examples=["elasticnet"],
    )
