# LEGATO MOBILE — TEAMMATE HANDOVER
**Last updated:** 2026-05-11
**Written by:** Nour Abulnasr (via Claude Code audit)
**For:** Teammate completing Flutter testing + final deployment

---

## 1. PROJECT OVERVIEW

Legato Mobile is the Flutter frontend for Legato — an AI legal ecosystem graduation project at Misr
International University (MIU), Faculty of Computer Science, AI track.

**What it does:**
- Upload Egyptian employment contracts (PDF/DOCX/image) → AI analysis with law violations
- Document chat: ask questions about a contract using the local LFM model
- Professional social network: feed, connections, profiles, endorsements
- 12 legal tools: explain clause, compare contracts, e-sign, negotiation coach, risk dashboard, etc.

**Stack:** Flutter + Provider + http package (no Dio, no GraphQL)
**Backend:** FastAPI + Python (separate repo: `https://github.com/nourabulnasr/GP-Legal-AI-`, branch `final_80%`)
**API base (default):** `http://10.0.2.2:8002` (Android emulator alias for host machine)

---

## 2. CURRENT STATUS — HONEST

### What is working and verified
- All 30 screens compile and run
- All API calls are wired to real backend — zero mock data
- `flutter analyze`: **0 errors, 0 warnings** (26 info-level style hints only)
- `flutter build apk --release`: **SUCCESS** (50.5 MB APK)
- `flutter build web`: **SUCCESS**
- All 8 critical bugs from the 2026-05-11 audit are fixed (see Section 4)

### What is NOT yet done
- **No real Android device test has been completed yet** — this is the main open task
- Feed screen UI is functional but visually generic (Material defaults, no polish)
- Network screen UI is functional but visually generic
- Git repo was only initialized on 2026-05-11 — no prior version history
- Backend is not yet deployed to Oracle Cloud or HF Spaces — APK currently points to emulator

---

## 3. WHAT IS DONE

### Authentication
| Feature | Screen | API |
|---------|--------|-----|
| Login | `auth/login_screen.dart` | POST `/auth/login` |
| Register | `auth/register_screen.dart` | POST `/auth/register` |
| Email verification | `auth/verify_email_screen.dart` | POST `/auth/verify-email` |
| Forgot password (3-step) | `auth/forgot_password_screen.dart` | Full reset flow |
| Session guard | `auth_gate.dart` | GET `/auth/me` on startup |
| 401 dialog (not silent crash) | `api/api_client.dart` | — |
| Foreground resume re-validation | `main.dart` _AppLifecycle | GET `/auth/me` |

### Contract Analysis
| Feature | Screen | API |
|---------|--------|-----|
| File upload (PDF/DOCX/image) | `analyze/analyze_screen.dart` | POST `/ocr_check_and_search` |
| Android 13+ file permissions | `AndroidManifest.xml` | — |
| Analysis results (violations, risk, RAG hits, OCR) | `history/analysis_detail_screen.dart` | — |
| Analysis history list | `history/history_screen.dart` | GET `/analyses` |
| Delete analysis | `history/history_screen.dart` | DELETE `/analyses/{id}` |
| WakeLock during analysis (prevents Doze kill) | `analyze/analyze_screen.dart` | — |
| 10-minute timeout for analysis | `config/app_config.dart` | — |

### Chat
| Feature | Screen | API | Model |
|---------|--------|-----|-------|
| General assistant | `chat/chat_assistant_screen.dart` | POST `/chat/assistant` | Gemini |
| Document chat (multi-turn) | `chat/chat_analysis_screen.dart` | POST `/chat/document` | **LFM (local)** |
| Document chat direct | `chat/chat_document_screen.dart` | POST `/chat/document` | **LFM (local)** |
| Analysis dropdown (no manual ID) | `chat/chat_analysis_screen.dart` | GET `/analyses` | — |
| Chat hub routing | `chat/chat_hub_screen.dart` | — | — |

### Social Feed
| Feature | Screen | API |
|---------|--------|-----|
| Infinite scroll feed | `social/feed_screen.dart` | GET `/api/posts?page=N` |
| Category filter chips | `social/feed_screen.dart` | — |
| Create post (composer sheet) | `social/feed_screen.dart` | POST `/api/posts` |
| Like / unlike | `social/feed_screen.dart` | POST `/api/posts/{id}/like` |
| Comments (expand, load, add) | `social/feed_screen.dart` | GET/POST `/api/posts/{id}/comments` |
| Share sheet (Copy / Browser / WhatsApp) | `social/feed_screen.dart` | POST `/api/posts/{id}/share` |
| Author click → member profile | `social/feed_screen.dart` | — |
| Timestamps in device local time | `social/feed_screen.dart` | — |

### Network
| Feature | Screen | API |
|---------|--------|-----|
| Connections stats grid | `social/network_screen.dart` | GET `/api/network/stats` |
| People you may know | `social/network_screen.dart` | GET `/api/network/suggestions` |
| Connect button (with sent-state) | `social/network_screen.dart` | POST `/api/network/invite` |
| Pending invitations badge + accept | `social/network_screen.dart` | GET/POST `/api/network/invitations` |
| Network search by name | `social/network_screen.dart` | GET `/api/network/search?q=` |
| Tap name → member profile | `social/network_screen.dart` | — |

### Profile
| Feature | Screen | API |
|---------|--------|-----|
| Own profile (cinematic cover + avatar) | `social/profile_screen.dart` | GET `/api/profile/{id}` |
| Edit profile dialog | `social/profile_screen.dart` | PUT `/api/profile` |
| Add education | `social/profile_screen.dart` | POST `/api/profile/education` |
| Member profile (view others) | `social/member_profile_screen.dart` | GET `/api/profile/{id}` |
| Skills & endorsements | `social/profile_skills_screen.dart` | GET `/api/profile/{id}/endorsements` |
| My documents (upload + list) | `social/profile_documents_screen.dart` | GET/POST `/api/profile/documents` |
| Recommendations (give + receive) | `social/profile_recommendations_screen.dart` | GET `/api/profile/{id}/recommendations` |

### 12 Legato Tools (`screens/features/phase5_screens.dart`)
| # | Tool | API |
|---|------|-----|
| 1 | E-sign (drawing canvas → base64) | POST `/legato/signatures` |
| 2 | Risk dashboard (badge + violations) | GET `/legato/risk/{id}` |
| 3 | Explain clause (LFM + RAG) | POST `/legato/explain-clause` |
| 4 | Summarize clauses | POST `/legato/summarize-clauses` |
| 5 | Compare two contracts | POST `/legato/compare` |
| 6 | Negotiation coach | POST `/legato/negotiation-chat` |
| 7 | Share analysis (public token) | POST `/legato/shares` |
| 8 | Deal threads (full CRUD) | GET/POST `/legato/deal-threads` |
| 9 | Timeline (milestones) | GET/POST `/legato/timeline` |
| 10 | Clause checker | POST `/legato/check-clause` |
| 11 | Chat assistant (routes to chat screen) | — |
| 12 | View shared analysis | GET `/legato/shares/public/{token}` |

### Other Screens
- `social/alerts_screen.dart` — timeline events + pending invitations
- `social/contracts_tab_screen.dart` — shortcuts to analyze + history
- `home/dashboard_tab.dart` — backend health ping + quick actions (Tab 0)
- `admin/admin_screen.dart` — user list, role change, analysis list (admin only)
- `settings/settings_screen.dart` — sign out + runtime API URL override
- `more/more_screen.dart` — Swagger link, settings shortcuts
- `roadmap/roadmap_screen.dart` — static feature roadmap

---

## 4. WHAT IS NOT DONE — PRIORITIZED

### CRITICAL (must be done before demo or submission)

#### C1 — Real device test (41-step checklist)
**This is the main remaining task.** See Section 8 for the full checklist.
The app has not been run on a physical Android device after the 2026-05-11 fixes.
Every blocker found in testing must be fixed before the examination.

#### C2 — Fix placeholder IP in network security config
**File:** `android/app/src/main/res/xml/network_security_config.xml`

The file currently has `192.168.1.105` as the allowed LAN IP for cleartext HTTP.
This must match the actual IP of the laptop running the backend.

**How to find your IP:**
```
# Windows
ipconfig | findstr "IPv4"
# Look for 192.168.x.x or 10.x.x.x
```

**Edit the file — change this line:**
```xml
<domain includeSubdomains="true">192.168.1.105</domain>
```
**To your actual laptop IP:**
```xml
<domain includeSubdomains="true">192.168.1.YOUR_ACTUAL_IP</domain>
```

Then rebuild the APK. Real device cannot connect without this fix.

#### C3 — Production APK build (after backend deploys)
Once the backend is live on Oracle Cloud or HF Spaces:
```bash
flutter build apk --release \
  --dart-define=API_BASE_URL=http://ORACLE_IP:8002 \
  --dart-define=SHARE_BASE_URL=http://ORACLE_IP:8002
```
Install on device: `adb install build/app/outputs/flutter-apk/app-release.apk`

#### C4 — Firebase web deployment (for browser demo link)
After the web build, deploy to Firebase. The committee gets a URL instead of needing a device.
See Section 7 for full steps.

---

### IMPORTANT (examiner will notice)

#### I1 — Feed screen visual polish
The social feed uses default Material cards. Works correctly but looks generic.
**Files to improve:** `lib/screens/social/feed_screen.dart` — specifically `_PostCard` widget.
**What to fix:** Replace Card with a custom container. Add more space between posts.
Improve typography hierarchy in the post body. The author section is the biggest win.

#### I2 — Network screen visual polish
The "People you may know" section uses plain ListTile cards.
**Files to improve:** `lib/screens/social/network_screen.dart` — the `_suggestions.map(...)` section.
**What to fix:** Custom card design for suggestion items. The stats grid (2×2) is already decent.

---

### NICE TO HAVE (only if time allows)

- Fade-in animations when feed posts load in
- Skeleton loading placeholders instead of spinner on feed
- Profile screen avatar upload (currently shows initials only)
- Analysis result screen: animated severity progress bars
- E-sign screen: replace manual analysis ID text field with a dropdown
- Loading state persists across screen rotation (currently resets)

---

## 5. HOW TO RUN LOCALLY (Chrome — fastest for iteration)

**Prerequisites:**
- Flutter SDK installed (version ^3.11.1)
- Backend running locally:
  ```bash
  # In the backend repo (GP-Legal-AI-, branch final_80%):
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
  ```

**Run in Chrome:**
```bash
cd C:\dev\legato_mobile1
flutter run -d chrome --dart-define=API_BASE_URL=http://localhost:8002
```

**Notes on Chrome dev:**
- File picker works differently in browser (no native picker, uses web file dialog)
- The app defaults `API_BASE_URL` to `http://10.0.2.2:8002` — this is the Android emulator IP,
  not valid for Chrome. Always pass `--dart-define=API_BASE_URL=http://localhost:8002` for web.
- CORS is already configured on the backend to allow localhost on any port (added in Session 3).

**Run on Android emulator:**
```bash
flutter run -d emulator-5554
# No dart-define needed — 10.0.2.2:8002 is the default and it works for emulators
```

**Run on real Android device (same WiFi as laptop):**
```bash
# Find your laptop IP first
ipconfig | findstr "IPv4"

flutter run -d <device-id> --dart-define=API_BASE_URL=http://192.168.1.YOUR_IP:8002
```

---

## 6. HOW TO BUILD APK

**Debug APK (for quick install, larger size):**
```bash
flutter build apk --debug \
  --dart-define=API_BASE_URL=http://192.168.1.YOUR_IP:8002
adb install build/app/outputs/flutter-apk/app-debug.apk
```

**Release APK (for actual demo — 50MB, optimized):**
```bash
# For LAN testing (real device + local backend):
flutter build apk --release \
  --dart-define=API_BASE_URL=http://192.168.1.YOUR_IP:8002 \
  --dart-define=SHARE_BASE_URL=http://192.168.1.YOUR_IP:8002

# For deployed backend (Oracle Cloud):
flutter build apk --release \
  --dart-define=API_BASE_URL=http://ORACLE_PUBLIC_IP:8002 \
  --dart-define=SHARE_BASE_URL=http://ORACLE_PUBLIC_IP:8002
```

APK location after build:
```
build/app/outputs/flutter-apk/app-release.apk
```

**Install via ADB:**
```bash
adb devices                    # confirm device is connected
adb install -r app-release.apk # -r replaces existing install
```

---

## 7. HOW TO BUILD WEB + DEPLOY TO FIREBASE

**Build:**
```bash
flutter build web \
  --dart-define=API_BASE_URL=http://ORACLE_PUBLIC_IP:8002
# Output: build/web/
```

**Deploy to Firebase Hosting:**
```bash
# One-time setup (if not already done):
npm install -g firebase-tools
firebase login
firebase init hosting   # select build/web as public directory, single-page app = yes

# Every deployment:
firebase deploy --only hosting
# → gives you: https://YOUR-PROJECT.web.app
```

**Result:** The committee can open a URL in any browser instead of installing the APK.
This is the easiest demo path if Oracle Cloud is running.

---

## 8. THE 41-STEP REAL DEVICE TEST CHECKLIST

Run this on a real Android phone (not emulator) with the backend running.
Check each box. Any failure = fix before exam.

### Auth (6 steps)
- [ ] 1. Register with a real email → receive verification email → enter code → account created
- [ ] 2. Login with correct credentials → HomeShell loads with Dashboard as first tab
- [ ] 3. Login with wrong password → error message shown, no crash
- [ ] 4. Logout from Settings → redirected to Login screen
- [ ] 5. Login again → session restores correctly
- [ ] 6. Lock phone screen for 10+ seconds, then unlock → app resumes on same screen (WakeLock test for analyze — less critical for auth)

### Contract Analysis (8 steps)
- [ ] 7. Tap Contracts tab → Upload button visible
- [ ] 8. Upload a PDF from device storage → file picker opens, file selectable (Android 13+ permissions test)
- [ ] 9. Analyze a PDF → loading indicator shows → result screen appears with violations list
- [ ] 10. Analyze a DOCX file → same result flow works
- [ ] 11. Open a violation → explanation text appears (LFM — may take 30-60 seconds, this is expected)
- [ ] 12. Tap History tab → saved analyses listed
- [ ] 13. Open a saved analysis from History → full results display
- [ ] 14. Delete an analysis → disappears from list, no crash

### Chat (5 steps)
- [ ] 15. Open Chat Hub → two options visible (Assistant + Document Chat)
- [ ] 16. Chat Assistant → type a general legal question → Gemini reply arrives
- [ ] 17. Document Chat → dropdown loads your saved analyses (no manual ID needed)
- [ ] 18. Document Chat → select an analysis → ask a question → LFM reply arrives (may take 30-90 seconds)
- [ ] 19. Multi-turn: send a follow-up question → context is preserved, reply makes sense

### Social Feed (6 steps)
- [ ] 20. Feed tab loads → posts visible, no spinner stuck
- [ ] 21. Scroll down → more posts load (infinite scroll)
- [ ] 22. Tap Like on a post → count increments immediately
- [ ] 23. Tap Comment → comments expand → type a comment → submit → comment appears
- [ ] 24. Tap Share → bottom sheet appears with Copy Link / Open in Browser / WhatsApp
- [ ] 25. Tap an author's name → opens their profile (not a crash)

### Create Post (2 steps)
- [ ] 26. Tap the "Share an update…" field → composer sheet opens
- [ ] 27. Type content, select category, post → new post appears in feed on refresh

### Network (5 steps)
- [ ] 28. Network tab loads → stats grid visible (Connections, Endorsements, etc.)
- [ ] 29. "People you may know" section shows user suggestions (not empty)
- [ ] 30. Tap Connect on a suggestion → button changes to "Sent ✓" immediately
- [ ] 31. Type in search bar → search runs → results appear
- [ ] 32. Tap a result name → opens their profile

### Profile (4 steps)
- [ ] 33. Profile tab loads → your name shown, cover gradient visible
- [ ] 34. Tap Edit Profile → dialog opens → change display name → save → name updates
- [ ] 35. Tap Skills & Endorsements → screen loads, no crash
- [ ] 36. Tap My Documents → screen loads → upload a document → appears in list

### Legato Tools (7 steps)
- [ ] 37. Open Features Hub → all tool cards visible
- [ ] 38. Explain Clause → enter a clause → result with law citation appears
- [ ] 39. Risk Dashboard → select a saved analysis → risk badge and violations appear
- [ ] 40. E-sign → draw a signature on canvas → tap submit → success message
- [ ] 41. Negotiation Coach → enter a clause → advice appears

### After completing all 41 steps
If any step fails, note the exact error message or behavior and fix it before the examination.
The most likely failure points on real devices are:
- Step 8 (file picker permissions) — check AndroidManifest.xml has READ_MEDIA_IMAGES
- Step 9/11/18 (LFM timeout) — the backend must be running with LFM loaded, first call always slow
- Step 29/30 (network suggestions empty) — check that the backend has demo data seeded

---

## 9. KNOWN VISUAL ISSUES

| Issue | Screen | Impact | Effort to fix |
|-------|--------|--------|---------------|
| Post cards are generic Material design | Feed | Visual quality only, no UX impact | Medium (2-3 hours) |
| Network suggestion cards are plain ListTiles | Network | Visual quality only | Small (1 hour) |
| `value:` deprecated warning on DropdownButtonFormField | `phase5_screens.dart:523` | Zero runtime impact, lint only | Trivial (rename to `initialValue:`) |
| E-sign screen uses manual analysis ID text input | E-sign | UX friction | Medium (add dropdown like ChatAnalysisScreen) |
| Share URL points to API path, not a web page | ShareFeatureScreen | Share link is not user-friendly | Needs web frontend to be deployed first |
| No loading state persistence on screen rotation | All screens | UX regression during orientation change | Medium |

---

## 10. KEY FILES MAP

```
C:\dev\legato_mobile1\
│
├── pubspec.yaml                        — dependencies
├── HANDOVER.md                         — this file
├── LEGATO_AGENT_CONTEXT.md             — full system architecture reference
├── LEGATO_FLUTTER_TASKS.md             — audit task list (historical)
│
├── android/
│   └── app/src/main/
│       ├── AndroidManifest.xml         — permissions (READ_MEDIA_IMAGES, etc.)
│       └── res/xml/
│           └── network_security_config.xml  ← EDIT THIS: change 192.168.1.105 to real IP
│
├── lib/
│   ├── main.dart                       — app entry, providers, lifecycle
│   ├── app_services.dart               — AppServices (ApiClient + LegatoApi)
│   ├── theme/linkedin_theme.dart       — all colors (gold: #C9A227, dark: #1B1F23)
│   │
│   ├── config/
│   │   ├── app_config.dart             — compile-time URL + timeout constants
│   │   └── runtime_config.dart         — SharedPreferences URL override (Settings screen)
│   │
│   ├── api/
│   │   ├── api_client.dart             — HTTP client (JWT injection, 401 hook, timeouts)
│   │   ├── legato_api.dart             — all typed API methods
│   │   └── api_exception.dart          — ApiException type
│   │
│   ├── providers/
│   │   └── auth_provider.dart          — AuthProvider (user, JWT, session expiry)
│   │
│   └── screens/
│       ├── auth_gate.dart              — route guard: loading / guest / logged-in
│       ├── auth/                       — login, register, verify_email, forgot_password
│       ├── home/
│       │   ├── home_shell.dart         — 6-tab NavigationBar (Home/Feed/Network/Contracts/Alerts/Profile)
│       │   └── dashboard_tab.dart      — Tab 0: health ping + quick actions
│       ├── analyze/
│       │   └── analyze_screen.dart     — contract upload + WakeLock + results
│       ├── history/
│       │   ├── history_screen.dart     — saved analyses list + delete
│       │   └── analysis_detail_screen.dart — full analysis view
│       ├── chat/
│       │   ├── chat_hub_screen.dart    — chat mode selector
│       │   ├── chat_assistant_screen.dart  — Gemini general chat
│       │   ├── chat_analysis_screen.dart   — LFM document chat with analysis dropdown
│       │   └── chat_document_screen.dart   — LFM document chat direct
│       ├── social/
│       │   ├── feed_screen.dart        — social feed ← NEEDS VISUAL POLISH
│       │   ├── network_screen.dart     — connections + suggestions ← NEEDS VISUAL POLISH
│       │   ├── profile_screen.dart     — own profile (cinematic design — good)
│       │   ├── member_profile_screen.dart  — view any user's profile
│       │   ├── profile_skills_screen.dart  — skills + endorsements
│       │   ├── profile_documents_screen.dart — upload + list legal docs
│       │   ├── profile_recommendations_screen.dart
│       │   ├── alerts_screen.dart      — timeline + pending invites
│       │   └── contracts_tab_screen.dart — shortcuts to analyze/history
│       ├── features/
│       │   ├── features_hub_screen.dart    — 12-tool grid
│       │   └── phase5_screens.dart         — all 12 tool screens
│       ├── admin/admin_screen.dart     — admin panel (role-gated)
│       ├── settings/settings_screen.dart — sign out + runtime URL override
│       ├── more/more_screen.dart       — Swagger + links
│       └── roadmap/roadmap_screen.dart — static roadmap
```

---

## 11. CREDENTIALS — WHERE TO CONFIGURE THE API URL

The app has **zero hardcoded passwords or keys** in Flutter. The only configuration needed is the
backend URL.

### Option A: dart-define at build time (for APK / web builds)
```bash
flutter build apk --dart-define=API_BASE_URL=http://YOUR_IP:8002
flutter build web  --dart-define=API_BASE_URL=http://YOUR_IP:8002
```

### Option B: Runtime override in Settings (no rebuild needed)
1. Open the app → go to Settings (bottom-right tab → gear icon → More screen → Settings)
2. Enter the new base URL: `http://192.168.1.YOUR_IP:8002`
3. Save → restart the app → the new URL is used

This is the fastest way to switch between local and deployed backend during testing.

### Backend credentials (backend repo only, not in Flutter)
The backend `.env` file contains `GEMINI_API_KEY` and `SECRET_KEY`.
Flutter does not need these — they are backend-only.

### Demo accounts (seeded by `seed_demo_data.py` in backend)
```
Admin:   admin@legato.com  /  LegatoAdmin2026!
Demo:    demo@legato.com   /  LegatoDemo2026!
```

---

## 12. TROUBLESHOOTING

### "Every API call fails on real device"
**Cause:** API URL `10.0.2.2:8002` is the Android emulator alias. Real devices cannot use it.
**Fix:** Build with your laptop's actual LAN IP (both device and laptop must be on same WiFi):
```bash
flutter build apk --debug --dart-define=API_BASE_URL=http://192.168.1.YOUR_IP:8002
```
Or use the Settings runtime override (Option B in Section 11).

### "File picker shows nothing / crashes on file pick"
**Cause:** Android 13+ requires `READ_MEDIA_IMAGES` and `READ_MEDIA_VIDEO` permissions.
**Verify they are in AndroidManifest.xml:**
```xml
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<uses-permission android:name="android.permission.READ_MEDIA_VIDEO" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />
```
These are already added. If it still fails, check that the device is actually Android 13+
and that you granted storage permission when prompted.

### "Document chat takes forever / times out"
**Expected behavior:** The LFM model runs on CPU. First response after server start takes
2-5 minutes (model loading). Subsequent responses: 30-90 seconds.
**What to do:**
1. Start the backend at least 5 minutes before the demo
2. Send a warm-up request: open Document Chat → select any analysis → send "hello"
3. Wait for the first response. All following responses will be faster.
4. If it times out (5-minute Flutter timeout), retry once.

### "CORS error in Chrome web dev"
**Cause:** Backend CORS config may not include your specific localhost port.
**Fix:** The backend has `allow_origin_regex` for all `localhost:\d+` ports.
Make sure you're running the backend from `GP-Legal-AI-` branch `final_80%` with the CORS fix
from session 3 (2026-05-10). If in doubt, use the emulator instead of Chrome for testing.

### "Network suggestions are empty"
**Cause:** Backend has no users except the one you're logged in as, or suggestions endpoint
returns a shape the Flutter code doesn't recognize.
**Fix 1:** Register 2-3 demo accounts → reconnect to backend → suggestions will populate.
**Fix 2:** Check backend logs — if `/api/network/suggestions` returns `{'suggestions': [...]}`,
Flutter handles it. If it returns something else, inspect the API response and update the
null-coalescing line in `network_screen.dart:53`.

### "Backend won't start / LFM not found"
The LFM model (2.23 GB) lives at `./LFM2.5-1.2B-Instruct/` in the backend repo.
It is too large for GitHub — it must be copied manually.
See `LEGATO_AGENT_CONTEXT.md` → Deployment Architecture for the `scp` command.

### "App stuck on loading spinner after login"
**Cause:** `GET /auth/me` is failing — backend is down or JWT is corrupted.
**Fix:** Go to Settings → sign out → sign in again. Or kill and restart the backend.

---

## 13. CONTACT

**Project owner:** Nour Abulnasr
**GitHub:** `github.com/nourabulnasr`
**Backend repo:** `https://github.com/nourabulnasr/GP-Legal-AI-` (branch: `final_80%`)
**Flutter repo:** `https://github.com/nourabulnasr/GP-Legal-AI-` (branch: `flutter-mobile`)
**Email:** 1432amr@gmail.com

If you find a bug that blocks a test step, fix it and commit to the `flutter-mobile` branch.
If you find a backend issue (API response format, missing endpoint, server error), flag it —
the backend may need a fix in the `final_80%` branch separately.

---

*End of handover. Last audit: 2026-05-11. All 8 critical Flutter bugs confirmed fixed.*
*The single most important remaining task: run the 41-step test on a real Android device.*
