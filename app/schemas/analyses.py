from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class AnalysisCreateRequest(BaseModel):
    filename: str = Field(default="contract.pdf")
    result_json: str = Field(..., description="Full JSON result of OCR+Check endpoint")

    # Optional document metadata
    mime_type: str | None = Field(default=None, description="e.g. application/pdf")
    sha256: str | None = Field(default=None, description="Hash of uploaded document (optional)")
    page_count: int | None = Field(default=None, description="Number of pages if PDF")
    ocr_used: int | None = Field(default=None, description="1 if OCR was used, else 0")
    detected_lang: str | None = Field(default=None, description="Detected language hint")


class AnalysisResponse(BaseModel):
    id: int
    filename: str | None = None
    created_at: datetime


class AnalysisDetailResponse(AnalysisResponse):
    user_id: int
    result_json: str


class AdminUpdateRoleRequest(BaseModel):
    role: Optional[str] = Field(None, description="Must be 'admin' or 'user'")
    user_type: Optional[str] = Field(None, description="'user' or 'lawyer'")
    lawyer_status: Optional[str] = Field(None, description="approved / rejected / pending")


class FlagRequest(BaseModel):
    needs_review: bool = Field(..., description="True = flag for lawyer review; False = clear flag")
    lawyer_note: str | None = Field(default=None, description="Optional note from reviewer")
