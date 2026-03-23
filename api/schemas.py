from pydantic import BaseModel, Field
from typing import List

class RecommendationRequest(BaseModel):
    n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return"
    )

class RecommendedItem(BaseModel):
    item_id : str
    score   : float

class RecommendationResponse(BaseModel):
    user_id         : str
    recommendations : List[RecommendedItem]
    model_version   : str
    served_from     : str
    total           : int

class HealthResponse(BaseModel):
    status        : str
    model_loaded  : bool
    redis_connected: bool