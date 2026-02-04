"""
HTML email templates matching the Legato frontend design.
Uses inline CSS for broad email client support.
Colors: primary #3b82f6, background #f5f5f5 / #ffffff, text #0c0c0f, muted #64748b.
"""


def verification_email_html(
    *,
    code: str,
    purpose: str,
    frontend_url: str,
    expire_minutes: int,
) -> str:
    """Build HTML for verification code email (signup or password_reset)."""
    if purpose == "signup":
        title = "Verify your email"
        sub = "Enter the 6-character code below on the verification page to complete signup."
        verify_path = "/verify-email"
    else:
        title = "Password reset code"
        sub = "Enter the code below on the password reset page to continue."
        verify_path = "/forgot-password/verify"
    verify_link = f"{frontend_url.rstrip('/')}{verify_path}"
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Legato – {title}</title></head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background-color:#f5f5f5;">
<div class="wrapper" style="max-width:480px;margin:0 auto;padding:24px 16px;">
  <div class="card" style="background:#ffffff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);padding:32px;">
    <div class="brand" style="font-size:18px;font-weight:600;color:#0c0c0f;margin-bottom:24px;">Legato</div>
    <h1 style="font-size:22px;font-weight:700;color:#0c0c0f;margin:0 0 8px 0;">{title}</h1>
    <p class="sub" style="font-size:14px;color:#64748b;margin:0 0 24px 0;line-height:1.5;">{sub}</p>
    <p style="font-size:14px;color:#0c0c0f;margin:0;">Your verification code:</p>
    <div class="code-box" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;font-family:ui-monospace,monospace;font-size:20px;letter-spacing:2px;color:#0c0c0f;margin:16px 0;">{code}</div>
    <p style="font-size:14px;color:#64748b;margin:0;">This code expires in {expire_minutes} minutes.</p>
    <p style="margin:20px 0 0 0;"><a href="{verify_link}" style="color:#3b82f6;font-size:14px;">Open verification page →</a></p>
    <p class="muted" style="font-size:13px;color:#64748b;margin-top:24px;line-height:1.5;">If you didn't request this, you can ignore this email.</p>
  </div>
  <p class="footer" style="font-size:12px;color:#94a3b8;margin-top:32px;">© 2025 Legato. All rights reserved.</p>
</div>
</body>
</html>"""


def reset_link_email_html(
    *,
    reset_link: str,
    frontend_url: str,
    expire_hours: int,
) -> str:
    """Build HTML for password reset link email."""
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Legato – Reset your password</title></head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background-color:#f5f5f5;">
<div class="wrapper" style="max-width:480px;margin:0 auto;padding:24px 16px;">
  <div class="card" style="background:#ffffff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);padding:32px;">
    <div class="brand" style="font-size:18px;font-weight:600;color:#0c0c0f;margin-bottom:24px;">Legato</div>
    <h1 style="font-size:22px;font-weight:700;color:#0c0c0f;margin:0 0 8px 0;">Reset your password</h1>
    <p class="sub" style="font-size:14px;color:#64748b;margin:0 0 24px 0;line-height:1.5;">You requested a password reset. Click the button below to set a new password.</p>
    <p style="margin:16px 0;"><a href="{reset_link}" class="btn" style="display:inline-block;background:#3b82f6;color:#ffffff !important;text-decoration:none;font-weight:500;font-size:14px;padding:10px 20px;border-radius:8px;">Reset password</a></p>
    <p style="font-size:14px;color:#64748b;margin:0;">This link expires in {expire_hours} hours.</p>
    <p class="muted" style="font-size:13px;color:#64748b;margin-top:24px;line-height:1.5;">If you didn't request this, ignore this email. Your password will stay the same.</p>
  </div>
  <p class="footer" style="font-size:12px;color:#94a3b8;margin-top:32px;">© 2025 Legato. All rights reserved.</p>
</div>
</body>
</html>"""
