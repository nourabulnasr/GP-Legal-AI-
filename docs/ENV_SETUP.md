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

## 2. SMTP (Password Reset Emails)

### Gmail

1. Enable 2-Step Verification: [Google Account → Security → 2-Step Verification](https://myaccount.google.com/security)
2. Create App Password: Security → 2-Step Verification → App passwords → Generate
3. Add to `.env`:
   ```
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=xxxx xxxx xxxx xxxx
   SMTP_FROM=your-email@gmail.com
   ```

### Other providers (Outlook, Yahoo, etc.)

- Check your provider’s SMTP settings
- Use the correct host, port (usually 587 for TLS), and credentials
- Set `SMTP_FROM` to your sender address

### Dev mode (no SMTP)

- If `SMTP_HOST` and `SMTP_USER` are empty, reset links are printed to the backend console

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
   GOOGLE_CLIENT_SECRET=xxx
   GOOGLE_REDIRECT_URI=http://127.0.0.1:8000/auth/google/callback
   ```

---

## 5. Frontend URL (for redirects)

```
FRONTEND_URL=http://localhost:5173
```

Use your actual frontend URL if it’s different.
