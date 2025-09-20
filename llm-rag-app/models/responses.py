from typing import List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health Status")
    database_connection: bool = Field(..., description="Database Connection Status")