# LEGATO MOBILE — HANDOVER DOCUMENT
Generated: 2026-05-11 | Full audit completed

---

## 1. PROJECT OVERVIEW

Legato Mobile is the Flutter frontend for Legato — an AI-powered legal ecosystem for Egyptian employment contracts. It is a graduation project at Misr International University (MIU), Faculty of Computer Science, AI track.

The app connects to a FastAPI backend running either locally (`http://10.0.2.2:8002` emulator default) or on a deployed server. All UI is real — no mock data. The backend provides 73 endpoints across 6 routers.

**Platform targets:** Android (primary), Flutter Web (secondary demo)
**Examiner context:** Committee will demo on real Android device + possibly web browser

---

## 2. ARCHITECTURE

### Entry Point
```
lib/main.dart
  └── RuntimeConfig.init()          — loads saved API URL override from SharedPreferences
  └── AuthProvider.bootstrap()      — checks saved JWT, calls GET /auth/me, navigates accordingly
  └── MaterialApp(home: AuthGate)
```

### AuthGate Flow
```
AuthGate (lib/screens/auth_gate.dart)
  ├── AuthState.loading   → loading spinner
  ├── AuthState.guest     → LoginScreen
  └── AuthState.loggedIn  → HomeShell (6-tab nav)
                              ├── [0] DashboardTab    — backend health + quick actions
                              ├── [1] FeedScreen      — social feed (posts/likes/comments/shares)
                              ├── [2] NetworkScreen   — connections, suggestions, search
                              ├── [3] ContractsTabScreen — upload + history
                              ├── [4] AlertsScreen    — timeline events + pending invites
                              └── [5] ProfileScreen   — own profile (cinematic redesign)
```

### State Management
- `AuthProvider` (ChangeNotifier) — holds User object, JWT, session expiry handling
- `AppServices` (plain Provider) — holds `ApiClient` + `LegatoApi`, wires 401 callback to AuthProvider

### API Layer
```
lib/api/api_client.dart    — base HTTP client (JWT injection, timeout, 401 interception)
lib/api/legato_api.dart    — typed API methods for every endpoint
lib/config/app_config.dart — compile-time URL configuration via --dart-define
lib/config/runtime_config.dart — SharedPreferences override for demo URL switching
```

---

## 3. WHAT'S DONE

### Auth (complete)
- Login (POST /auth/login) ✓
- Register (POST /auth/register) ✓
- Email verification (POST /auth/verify-email) ✓
- Forgot password 3-step flow ✓
- Session expiry dialog on 401 (not silent crash) ✓
- App lifecycle observer: re-validates JWT on foreground resume ✓

### Contract Analysis (complete)
- File picker (PDF/DOCX/image) with Android 13+ permissions ✓
- POST /ocr_check_and_search with 10-minute timeout ✓
- Full result display: rule violations, risk score, OCR text, RAG hits ✓
- Analysis history (list + delete) ✓
- Analysis detail screen ✓
- WakeLock during analysis (prevents Doze killing long requests) ✓

### Chat (complete)
- Chat hub (hub screen with all chat entry points) ✓
- Chat assistant (POST /chat/assistant via Gemini) ✓
- Chat with document (POST /chat/document via LFM) ✓
  - Dropdown loads saved analyses from GET /analyses — no manual ID required ✓
- Document chat screen (POST /chat/document, multi-turn) ✓

### Social Feed (complete)
- Infinite scroll feed with pagination ✓
- Category filter chips ✓
- Create post composer (bottom sheet, with category + tag selection) ✓
- Like/unlike toggle ✓
- Comments (load on expand, add comment) ✓
- Share sheet (native: copy link / open browser / WhatsApp — no share_plus dependency) ✓
- Author name click → MemberProfileScreen ✓
- Timestamps: UTC → device local time via `.toLocal()` ✓

### Network (complete)
- Stats grid (connections, endorsements, profile views, invitations) ✓
- People you may know suggestions (handles `items` + `suggestions` response keys) ✓
- Connect button with sent-state feedback (`_sentInvites` Set) ✓
- Pending invitations badge + accept flow ✓
- Network search (real GET /api/network/search) ✓
- Tap name → MemberProfileScreen ✓

### Profile (complete — cinematic redesign)
- SliverAppBar with dark navy-to-gold gradient cover (expandedHeight: 160) ✓
- Overlapping avatar with white border + shadow ✓
- Stats pills (connections, endorsements, views) ✓
- Edit profile dialog (name, title, company, location, bio, skills) ✓
- Sections: About, Skills, Experience, Education ✓
- Add Education dialog ✓
- Links to: Skills & Endorsements, My Documents, Recommendations ✓

### 12 Legato Tools (phase5_screens.dart — all implemented)
1. E-sign — hand_signature canvas → base64 → POST /legato/signatures ✓
2. Risk dashboard — parses JSON → severity badge + stat chips + top-3 violations ✓
3. Explain clause — POST /legato/explain-clause (LFM + RAG) ✓
4. Summarize clauses — POST /legato/summarize-clauses ✓
5. Compare contracts — dual file picker → POST /legato/compare ✓
6. Negotiation coach — POST /legato/negotiation-chat ✓
7. Share analysis — POST /legato/shares → public token link ✓
8. Deal threads — full CRUD /legato/deal-threads ✓
9. Timeline — GET/POST /legato/timeline ✓
10. Clause checker — POST /legato/check-clause ✓
11. Chat assistant (routed to ChatAssistantScreen) ✓
12. Public share — view via share token ✓

### Other Screens
- Member profile (view any user) ✓
- Skills & Endorsements (view + request endorsement) ✓
- Profile Documents (upload + list contracts) ✓
- Recommendations (give + receive) ✓
- Alerts (timeline events + pending invitations) ✓
- Contracts tab (shortcut to analyze/history) ✓
- Admin panel (users + analyses — role-guarded client-side) ✓
- Settings (sign out + API URL override for demo switching) ✓
- More screen (links to Swagger docs, roadmap) ✓
- Roadmap (static feature roadmap view) ✓
- Dashboard tab (backend health ping + quick actions) ✓

---

## 4. WHAT'S NOT DONE (honest list)

### Partially built or known limitations

| Item | Location | Status |
|------|----------|--------|
| Analysis picker in E-sign | `phase5_screens.dart` — EsignFeatureScreen | Still uses manual analysis ID text input (dropdown not added here, only in chat_analysis_screen) |
| Share URL uses internal API path | `phase5_screens.dart` — ShareFeatureScreen | URL is the API endpoint, not a web frontend page. Needs real domain. |
| Profile camelCase/snake_case mix | `legato_api.dart` — `putProfileResilient` | Sends `displayName` (camelCase), reads both — may silently lose name if backend enforces snake_case only |
| JWT stored unencrypted | `lib/providers/auth_provider.dart` + SharedPreferences | `flutter_secure_storage` was removed due to Windows build path issue. Token stored in plain SharedPreferences. |
| No deep linking / go_router | All navigation | Uses `Navigator.push` + `MaterialPageRoute` only. No named routes. Share links cannot open app to specific screen. |
| Named routing | All screens | All push-based. Back stack can behave unexpectedly in nested flows. |
| `value` deprecated in DropdownButtonFormField | `phase5_screens.dart:523` | `value:` → `initialValue:` (info-level lint only, works fine at runtime) |
| No real HTTPS / prod TLS | `network_security_config.xml` | Scoped to dev IPs. For production, add server's public domain. |
| Admin endpoints not server-enforced client-side | `admin_screen.dart` | Hidden by `user.role == 'admin'` check but a guessed route exposes admin actions. Backend enforces — client doesn't block the route. |

---

## 5. HOW TO RUN LOCALLY

### Prerequisites
- Flutter SDK (version ^3.11.1 per pubspec.yaml)
- Android emulator or physical device
- Backend running: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8002`

### Flutter Web (dev)
```bash
cd C:\dev\legato_mobile1
flutter run -d chrome --dart-define=API_BASE_URL=http://localhost:8002
```

### Android Emulator (dev)
```bash
flutter run -d emulator-5554
# Default API_BASE_URL=http://10.0.2.2:8002 works for emulator
```

### Android Real Device (dev)
```bash
# Find your PC's LAN IP (e.g. 192.168.1.45)
flutter run -d <device-id> --dart-define=API_BASE_URL=http://192.168.1.45:8002
```

### APK Release Build
```bash
flutter build apk --release \
  --dart-define=API_BASE_URL=http://YOUR_SERVER_IP:8002 \
  --dart-define=SHARE_BASE_URL=http://YOUR_SERVER_IP:8002
# Output: build/app/outputs/flutter-apk/app-release.apk (approx 50MB)
```

### Flutter Web Release Build
```bash
flutter build web \
  --dart-define=API_BASE_URL=http://YOUR_SERVER_IP:8002
# Output: build/web/ → deploy to Firebase Hosting
```

### Runtime API URL Override (no rebuild needed)
Go to Settings screen in the app → enter the new base URL → save. Persists across restarts via SharedPreferences. Useful for demo switching between local and deployed backend.

---

## 6. API DEPENDENCIES

The app has no offline mode. Every feature requires the backend.

| Backend | Default address | Notes |
|---------|-----------------|-------|
| Local dev (emulator) | `http://10.0.2.2:8002` | Android emulator alias for host machine |
| Local dev (real device) | `http://<LAN_IP>:8002` | Must be same WiFi network |
| Deployed (HF Space) | `https://nourabulnasr-legato.hf.space` | Set via --dart-define at build time or runtime override |
| Deployed (Oracle A1) | `http://<ORACLE_IP>:8002` | Oracle Cloud always-free tier |

Backend repo: `https://github.com/nourabulnasr/GP-Legal-AI-` (branch: `final_80%`)

LFM model (2.23GB) must be on server at `./LFM2.5-1.2B-Instruct/`. Document chat and explain-clause will fail without it.

---

## 7. KEY FILES MAP

```
lib/
├── main.dart                          — app entry, providers, lifecycle observer
├── app_services.dart                  — AppServices (ApiClient + LegatoApi factory)
├── theme/linkedin_theme.dart          — all colors/typography (gold: #C9A227, dark: #1B1F23)
│
├── config/
│   ├── app_config.dart               — compile-time URL + timeout constants
│   └── runtime_config.dart           — SharedPreferences URL override
│
├── api/
│   ├── api_client.dart               — base HTTP (postJson, getJson, multipart, 401 hook)
│   ├── legato_api.dart               — all typed API methods (analyzeContract, chatWithDocument, etc.)
│   └── api_exception.dart            — ApiException with message field
│
├── providers/
│   └── auth_provider.dart            — AuthProvider (user, JWT, login/logout, session expiry)
│
└── screens/
    ├── auth_gate.dart                — route guard (loading/guest/loggedIn)
    ├── auth/                         — login, register, verify_email, forgot_password
    ├── home/
    │   ├── home_shell.dart           — 6-tab NavigationBar shell
    │   └── dashboard_tab.dart        — health ping + quick actions
    ├── analyze/analyze_screen.dart   — contract upload + result display
    ├── history/
    │   ├── history_screen.dart       — saved analyses list
    │   └── analysis_detail_screen.dart — full analysis view
    ├── chat/
    │   ├── chat_hub_screen.dart      — hub routing to all chat modes
    │   ├── chat_assistant_screen.dart — Gemini general chat
    │   ├── chat_analysis_screen.dart — LFM document chat (POST /chat/document)
    │   └── chat_document_screen.dart — direct document chat entry
    ├── social/
    │   ├── feed_screen.dart          — social feed (posts/likes/comments/shares)
    │   ├── network_screen.dart       — connections, suggestions, search
    │   ├── profile_screen.dart       — own profile (cinematic SliverAppBar design)
    │   ├── member_profile_screen.dart — any user's profile
    │   ├── profile_skills_screen.dart — skills + endorsements
    │   ├── profile_documents_screen.dart — upload + list legal documents
    │   ├── profile_recommendations_screen.dart — give/receive recommendations
    │   ├── alerts_screen.dart        — timeline + invitations
    │   └── contracts_tab_screen.dart — contracts shortcut tab
    ├── features/
    │   ├── features_hub_screen.dart  — grid of all 12 tools
    │   └── phase5_screens.dart       — all 12 tool screens (e-sign, risk, explain, etc.)
    ├── admin/admin_screen.dart       — admin panel (role-gated)
    ├── settings/settings_screen.dart — sign out + runtime API URL override
    ├── more/more_screen.dart         — Swagger link + settings shortcuts
    └── roadmap/roadmap_screen.dart   — static roadmap view
```

---

## 8. TESTING

### Real Device Test
No formal `docs/REAL_DEVICE_TEST.md` exists yet. Run this checklist manually:

**Auth flow (5 steps)**
1. Register with real email → receive verification email → paste code
2. Login with registered credentials
3. Login with wrong password → see error (not crash)
4. Logout → confirm redirected to login
5. Foreground app after 12+ hours → session expiry dialog appears (not silent redirect)

**Contract analysis (8 steps)**
6. Upload PDF from device storage → analyze → results appear
7. Upload DOCX → analyze → results appear
8. View violation detail → explanation text (LFM) present
9. Save analysis → appears in History
10. Open saved analysis from History
11. Delete analysis from History
12. Run Explain Clause on a violation
13. Run Summarize Clauses

**Chat (4 steps)**
14. Open Chat Hub → select Assistant → send message → Gemini replies
15. Open Document Chat → select analysis from dropdown → send question → LFM replies
16. Multi-turn: ask follow-up → context preserved
17. Error state: disconnect network → graceful error message

**Social (7 steps)**
18. Create a post with tags and category
19. Like a post → count updates
20. Expand comments → add a comment
21. Tap share → copy link → paste and verify URL format
22. Tap author name → opens their profile
23. Network tab → "Connect" button → "Sent ✓" appears
24. Network search → find a user by name

**Features (8 steps)**
25. E-sign: draw signature → submit → success message
26. Risk dashboard: select analysis → risk badge + violations appear
27. Compare: upload two contracts → comparison text appears
28. Negotiation coach: enter clause → advice appears
29. Share analysis → token link copied
30. Deal thread: create thread → add message
31. Timeline: view entries
32. Admin (admin account only): view users, change role

**Profile (3 steps)**
33. Edit profile → save → changes reflected immediately
34. Add education entry → appears in list
35. Navigate to Skills & Endorsements

**Settings (3 steps)**
36. Settings → change API URL → save → restart → new URL persists
37. Sign out → redirected to login
38. Sign back in → previous session data loaded

**Known issues during testing**
- LFM responses (explain-clause, chat/document) can take 30-120 seconds on CPU backend. This is expected. Phone must not sleep — the WakeLock helps but cannot override manual screen-off.
- If backend is not running: every API call shows an error message. This is expected behavior.
- `value:` deprecation warning on DropdownButtonFormField in phase5_screens.dart — cosmetic only, works at runtime.

---

## 9. PRODUCTION BUILD CHECKLIST

When the backend is deployed (Oracle Cloud or HF Space):

```bash
# 1. Get the backend's public IP or domain
# 2. Build APK
flutter build apk --release \
  --dart-define=API_BASE_URL=http://ORACLE_IP:8002 \
  --dart-define=SHARE_BASE_URL=http://ORACLE_IP:8002

# 3. Install on device
adb install build/app/outputs/flutter-apk/app-release.apk

# 4. Build web
flutter build web \
  --dart-define=API_BASE_URL=http://ORACLE_IP:8002

# 5. Deploy web to Firebase
firebase deploy --only hosting
# → legato.web.app (or custom domain)
```

**Before submitting to Play Store (future):**
- [ ] Update `android/app/src/main/res/xml/network_security_config.xml` with production domain
- [ ] Remove `10.0.2.2` and `192.168.` dev entries from network config
- [ ] Replace `SharedPreferences` JWT with `flutter_secure_storage` (fix Windows path issue first)
- [ ] Add `CAMERA` permission to AndroidManifest if camera features are added
- [ ] Set `minSdkVersion 21` and `targetSdkVersion 34` in android/app/build.gradle

---

## 10. FLUTTER ANALYZE RESULTS (2026-05-11)

```
flutter analyze: 26 issues found
  Errors:   0
  Warnings: 0
  Info:     26 (all cosmetic — null-aware syntax suggestions + 1 deprecated param)
```

Zero errors. Zero warnings. Build is clean.

---

## 11. DEPENDENCIES (pubspec.yaml)

| Package | Version | Purpose |
|---------|---------|---------|
| `http` | ^1.2.2 | HTTP client for all API calls |
| `provider` | ^6.1.2 | State management (AuthProvider, AppServices) |
| `shared_preferences` | ^2.3.3 | JWT + runtime URL persistence |
| `file_picker` | ^8.1.4 | Contract file selection (PDF/DOCX/image) |
| `url_launcher` | ^6.3.2 | Share sheet external links (WhatsApp, browser) |
| `hand_signature` | ^3.1.0+2 | E-sign drawing canvas |
| `wakelock_plus` | ^1.2.10 | Prevents screen sleep during long analysis |
| `cupertino_icons` | ^1.0.8 | iOS-style icon set |

---

*This handover document was generated via full codebase audit on 2026-05-11.*
*All 8 fixes from the 2026-05-11 session are confirmed in place.*
