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

class ClauseCheckResponse(BaseModel):
    clause_text: str
    language: str = "ar"
    matches: List[RuleHit] = []
