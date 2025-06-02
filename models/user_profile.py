from pydantic import BaseModel, Field
from typing import Optional

class UserProfile(BaseModel):
    income: float = Field(..., description="Monthly income")
    expenses: float = Field(..., description="Monthly expenses")
    debt: float = Field(default=0, description="Total debt")
    age: int = Field(..., description="Current age")
    assets: float = Field(default=0, description="Current assets")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance level")
    financial_goals: Optional[str] = Field(None, description="Financial goals")