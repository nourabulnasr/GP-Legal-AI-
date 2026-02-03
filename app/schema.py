from pydantic import BaseModel, Field
from typing import List, Optional

class HealthResponse(BaseModel):
    ok: bool = True
    service: str = "legalai"
    version: str = "0.1.0"

class OCRResponse(BaseModel):
    text: str

class ClauseCheckRequest(BaseModel):
    clause_text: str = Field(..., description="Raw clause text to analyze")
    law_scope: Optional[List[str]] = Field(default=None, description="e.g., ['labor','civil','commercial']")
    language: Optional[str] = None   # <-- NEW: lets the handler read req.language safely
    use_ml: Optional[bool] = False

class RuleHit(BaseModel):
    rule_id: str
    law: Optional[str] = None
    article: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    rationale: Optional[str] = None
    article_text: Optional[str] = None
    suggestion: Optional[str] = None
    suggestion_ref: Optional[str] = None

    # ML overlay (if available)
    ml_predicted: Optional[bool] = None
    ml_score: Optional[float] = None
    ml_passed_threshold: Optional[bool] = None

class ClauseCheckResponse(BaseModel):
    clause_text: str
    language: str = "ar"
    matches: List[RuleHit] = []

class MLPrediction(BaseModel):
    rule_id: str
    score: float
    passed_threshold: bool

class ClauseCheckResponseWithML(ClauseCheckResponse):
    ml_used: Optional[bool] = False
    ml_predictions: Optional[List[MLPrediction]] = None
    # Unified rule+ML pipeline: clause-level violation risk (0–1, threshold)
    unified_ml_risk: Optional[float] = None
    unified_ml_above_threshold: Optional[bool] = None    
