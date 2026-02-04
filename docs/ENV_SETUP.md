# Environment Setup — Step-by-Step

## 1. Create your `.env` file

```bash
# From backend root (GP-Legal-AI--main)
copy .env.example .env
# Or on Mac/Linux:
# cp .env.example .env
```

Edit `.env` with your values.

---

## 2. SMTP (Verification & Password Reset Emails)

**Where SMTP is read:** The backend loads `.env` from the **project root** (same folder as `app/`). Verification emails (signup and "Resend code") and password-reset emails all use these settings. For Gmail, the backend automatically uses `SMTP_USER` as the "From" address so messages are accepted.

### Gmail

1. Enable 2-Step Verification: [Google Account → Security → 2-Step Verification](https://myaccount.google.com/security)
2. Create App Password: Security → 2-Step Verification → App passwords → Generate
3. Add to **project root** `.env` (not the frontend `.env`):
   ```
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=xxxx xxxx xxxx xxxx
   SMTP_FROM=your-email@gmail.com
   ```
   For Gmail, `SMTP_FROM` is ignored and `SMTP_USER` is used as From. Ensure `SMTP_USER` and `SMTP_PASSWORD` (App Password) are correct.

### Other providers (Outlook, Yahoo, etc.)

- Check your provider's SMTP settings
- Use the correct host, port (usually 587 for TLS), and credentials
- Set `SMTP_FROM` to your sender address

### Dev mode (no SMTP)

- If `SMTP_HOST` and `SMTP_USER` are empty, verification codes and reset links are printed to the backend console. You can still use "Resend code" on the login or register verify step; if SMTP is not configured, the backend will return 503 and you’ll need to configure SMTP in the **project root** `.env`.

---

## 3. Chat (Gemini API)

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create or select an API key
3. Add to `.env`:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```
4. Restart the backend

If `GEMINI_API_KEY` is not set, the chat page will show: *"Set GEMINI_API_KEY to enable chat."*

---

## 4. Google OAuth (SSO Login)

1. Go to [Google Cloud Console → APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials)
2. Create OAuth 2.0 Client ID (Web application)
3. Add redirect URI: `http://127.0.0.1:8000/auth/google/callback` (or your backend URL)
4. Add to `.env`:
   ```
   GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
  GOOGLE_CLIENT_SECRET=your-client-secret-here
   GOOGLE_REDIRECT_URI=http://127.0.0.1:8000/auth/google/callback
   ```
   Use the **Client ID** and **Client secret** from the same OAuth 2.0 Client ID credentials card (not the Gemini API key). The callback reads these at request time from the environment.

---

## 5. Frontend URL (for redirects)

```
FRONTEND_URL=http://localhost:5173
```

Use your actual frontend URL if different.

---

## 6. Document AI (OCR on upload)

When you upload a PDF for analysis, the backend can use **Google Document AI** for better OCR (in `app/DocumentAI.py`). The pipeline in `app/main.py` prefers Document AI when it is configured.

To enable it, set in `.env`:

```
DOCUMENT_AI_PROJECT_ID=your-gcp-project-id
DOCUMENT_AI_LOCATION=us
DOCUMENT_AI_PROCESSOR_ID=your-processor-id
```

Create a Document AI processor in Google Cloud Console and use its project ID, location, and processor ID. If these are not set, the backend falls back to PyMuPDF/tesseract for PDF text extraction.
