from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    user_type: str = Field(default="user", pattern="^(user|lawyer)$")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    id: int
    email: str
    role: str = "user"
    user_type: str = "user"
    lawyer_status: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=6)


class VerifyResetCodeRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=1)


class VerifyResetCodeResponse(BaseModel):
    reset_token: str


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=1)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class LawyerApplicationResponse(BaseModel):
    id: int
    user_id: int
    user_email: str
    bar_license_number: Optional[str]
    document_filename: Optional[str]
    has_document: bool
    status: str
    admin_note: Optional[str]
    created_at: datetime
    reviewed_at: Optional[datetime]

    class Config:
        from_attributes = True


class AdminReviewLawyerRequest(BaseModel):
    action: str = Field(..., pattern="^(approve|reject)$")
    admin_note: Optional[str] = None
