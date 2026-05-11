# LEGATO FLUTTER — AUDIT & TASK LIST
Generated: 2026-05-08 | Criticals fixed: 2026-05-09

## ✅ ALL F-CRITICAL ITEMS FIXED (2026-05-09)
- F-C1: Android 13+ permissions added to AndroidManifest.xml ✓
- F-C3: DashboardTab wired as first tab in HomeShell (6-tab nav) ✓
- F-C4: Session expiry AlertDialog on 401 — no more silent redirect ✓
- F-C5: readOnly feed search bar removed entirely ✓
- F-C6: Risk screen parses JSON → severity badge + stat chips + top-3 violations ✓
- F-C7: hand_signature 3.1.0+2 added — drawing canvas → base64 → POST /legato/signatures ✓
- F-C8: ChatAnalysisScreen loads analyses dropdown (GET /analyses) — no manual ID ✓
- F-C9: Share URL uses AppConfig.shareBaseUrl (SHARE_BASE_URL dart-define) ✓
- F-C10: network_security_config.xml created — cleartext scoped to dev IPs, global flag removed ✓

## ✅ ALL F-MEDIUM AND F-POLISH ITEMS FIXED (2026-05-09 session 2)
- F-M1: VoiceAssistantFeatureScreen removed from FeaturesHub → ChatAssistantScreen wired ✓
- F-P1: BiometricInfoScreen tile removed from FeaturesHub grid (dead stub) ✓
- F-M4: FlutterError.onError + PlatformDispatcher.onError in main.dart ✓
- F-M3: API URL override in Settings screen — RuntimeConfig + SharedPreferences ✓
- F-M5: WakeLock during analysis — wakelock_plus 1.5.2 wraps analyzeContract ✓
- F-6: network_security_config.xml — subnet IPs corrected, comment added for demo ✓
- F-7: hand_signature deprecated API updated to SignaturePathSetup + ShapeSignatureDrawer ✓
- F-8: mounted check after async gap in EsignFeatureScreen ✓

## BUILD STATUS (2026-05-09 session 2)
- flutter analyze: 34 issues (1 warning, 33 info) — ZERO errors ✓
- flutter build apk --release: ✓ SUCCESS (336s, 50.5MB)
- New package: wakelock_plus 1.5.2 (resolved cleanly)
- New file: lib/config/runtime_config.dart

## REAL DEVICE DEPLOY COMMAND
flutter build apk --release --dart-define=API_BASE_URL=http://ORACLE_IP:8000 --dart-define=SHARE_BASE_URL=http://ORACLE_IP:8000

---

## PROJECT SUMMARY

- **42 Dart files** across auth, social, contracts, chat, features, admin
- **Stack:** Flutter + Provider + `http` package (no Dio)
- **Auth:** JWT via SharedPreferences → Bearer header
- **API base:** `http://10.0.2.2:8002` (Android emulator default)
- **All API calls are real** — no mock data found

---

## 🔴 CRITICAL — App crashes or core flow broken

### C1 — File picker will FAIL on Android 13+ (API 33+)
**File:** `android/app/src/main/AndroidManifest.xml`

Missing permissions: `READ_MEDIA_IMAGES`, `READ_MEDIA_VIDEO`, `READ_MEDIA_VISUAL_USER_SELECTED`

`file_picker` is used in `analyze_screen.dart`, `compare` feature, and `profile_documents_screen.dart`.
Without these declarations, file picker will silently fail or throw a permission exception on any Android 13+ device.

**Fix:** Add to AndroidManifest.xml:
```xml
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<uses-permission android:name="android.permission.READ_MEDIA_VIDEO" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />
```

---

### C2 — Real device cannot reach the API (emulator localhost)
**File:** `lib/config/app_config.dart`

Default API URL is `http://10.0.2.2:8002` — this is the Android emulator alias for host machine localhost. A **real Android device** on the same network cannot use this address. The app will show API errors for every single call.

**Fix:** Before testing on real device, build with:
```
flutter build apk --dart-define=API_BASE_URL=http://<YOUR_PC_LOCAL_IP>:8002
```
Or set a production URL for release builds.

---

### C3 — No JWT refresh — users silently logged out
**File:** `lib/api/api_client.dart`, `lib/providers/auth_provider.dart`

When the JWT token expires, the next API call returns 401. The app clears the token and bounces back to login — **mid-session, with no warning**. There is no refresh token mechanism at all.

**Impact:** Any user session longer than the token TTL will crash to login unexpectedly. Examiners testing the app for more than one session will hit this.

**Fix (minimum viable):** Show an explicit "Session expired — please log in again" dialog before redirecting. Long-term: implement refresh token endpoint on backend.

---

### C4 — `usesCleartextTraffic=true` blocks real HTTPS deployment
**File:** `android/app/src/main/AndroidManifest.xml`

Currently set globally to allow plain HTTP (needed for emulator dev). This must be scoped before any production or demo deployment, otherwise Android will block HTTPS upgrades and the flag is a Play Store rejection risk.

**Fix for production:** Create `android/app/src/main/res/xml/network_security_config.xml`:
```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">10.0.2.2</domain>
        <domain includeSubdomains="true">192.168.x.x</domain> <!-- dev LAN -->
    </domain-config>
</network-security-config>
```
Remove global `usesCleartextTraffic="true"` and reference this file in the manifest.

---

### C5 — DashboardTab widget is orphaned (dead code)
**File:** `lib/screens/home/dashboard_tab.dart`

A fully built dashboard screen with backend health ping exists but is **never mounted** anywhere. The `HomeShell` bottom nav goes directly to `FeedScreen`. The tab is invisible to users and examiners. This is likely meant to be the landing tab.

**Fix:** Wire `DashboardTab` as the first tab in `HomeShell`'s bottom nav, replacing or supplementing the current feed-first layout.

---

## 🟡 MEDIUM — Features missing or broken

### M1 — Feed search bar is permanently non-functional
**File:** `lib/screens/social/feed_screen.dart`

Search field is `readOnly: true` and shows a SnackBar: "filter by query param when API adds it." An examiner testing the search bar will see this immediately.

**Fix:** Either implement search via GET `/api/posts?q=...` or hide the search bar entirely until the endpoint is ready.

---

### M2 — Risk Dashboard shows raw JSON, not a visual risk score
**File:** `lib/screens/features/phase5_screens.dart` — `RiskFeatureScreen`

`GET /legato/risk/{id}` response is rendered as a raw JSON blob. No risk score visualisation, no color-coded severity, no chart.

**Fix (minimum):** Parse the response and display at minimum: overall risk level (badge/chip), top 3 risk items as a list.

---

### M3 — E-sign screen has no signature canvas
**File:** `lib/screens/features/phase5_screens.dart` — `EsignFeatureScreen`

Records only signer name + checkbox. There is no drawing canvas, no PDF preview, no cryptographic signature. The screen itself notes "connect DocuSign/Adobe in later phase."

**Fix (minimum viable for demo):** Add `signature` or `hand_signature` Flutter package and render a drawing canvas. POST the base64 image to `/legato/signatures`.

---

### M4 — ChatAnalysisScreen requires manual analysis ID input
**File:** `lib/screens/chat/chat_analysis_screen.dart`

When accessed from the Chat Hub (not from AnalysisDetailScreen), the user must manually type an analysis ID. End-users don't know their analysis IDs.

**Fix:** Add a dropdown/picker that loads analysis history via GET `/analyses` and lets user select by contract name.

---

### M5 — Share link points to internal API URL, not a public web URL
**File:** `lib/screens/features/phase5_screens.dart` — `ShareFeatureScreen`

The generated share link is: `http://10.0.2.2:8002/legato/shares/public/{token}` — this is not accessible to anyone outside the dev machine. Even on a real server it points to the raw API endpoint, not a user-facing web page.

**Fix:** Build the share URL as a web frontend URL, e.g.: `https://legato.app/shared/{token}`. If no web frontend exists, at minimum use the real server's public domain.

---

### M6 — Profile API uses mixed camelCase/snake_case — silent field loss
**File:** `lib/api/legato_api.dart` — `putProfileResilient` + `_mapLegalProfileToSocialShape`

`displayName` is sent as camelCase but the mapper reads both `displayName` and `display_name`. If the backend enforces one convention, profile edits will appear to save but silently lose the name.

**Fix:** Align with the actual backend schema — pick one convention and remove the dual-key fallback.

---

### M7 — Settings screen has no real settings
**File:** `lib/screens/settings/settings_screen.dart`

Only contains: sign out button + biometrics info screen. No notification preferences, no API URL override for production testing, no theme toggle, no language setting.

**Fix (for demo):** Add at minimum: API URL override input (for switching between local/production) and a "Clear local data" option.

---

### M8 — Voice Assistant duplicates Chat Assistant
**File:** `lib/screens/features/phase5_screens.dart` — `VoiceAssistantFeatureScreen`

This screen is identical to `ChatAssistantScreen` with voice UI stripped out (speech-to-text removed). It appears twice in navigation — once in the Features Hub and once would be accessible from More screen. Confusing duplication.

**Fix:** Remove `VoiceAssistantFeatureScreen` from the Features Hub grid and link directly to `ChatAssistantScreen`. Or restore voice input.

---

### M9 — No named routing — back-stack fragile, deep linking impossible
**Architecture issue — all screens**

All navigation uses `Navigator.push(MaterialPageRoute(...))`. No `go_router` or named routes. This means:
- No deep links (share links that open the app to a specific screen)
- Back button behavior unpredictable in nested feature flows
- No way to share a URL that opens a specific analysis

**Fix:** Migrate to `go_router` (or at minimum define named routes in `MaterialApp.routes`). Priority: share links and analysis detail deep links.

---

## 🟢 POLISH

### P1 — JWT stored in SharedPreferences (not encrypted)
**File:** `lib/storage/token_storage.dart`

The file itself documents this gap — `flutter_secure_storage` was removed due to a Windows build path issue (`C:\Users\Aly ahmed\` space in path breaks native hooks). For production, the token must be encrypted.

**Fix:** After resolving the path issue (rename user folder or use a symlink), replace `SharedPreferences` with `flutter_secure_storage`.

---

### P2 — No global error boundary
No `FlutterError.onError` handler or global Snackbar scaffold. Unhandled widget exceptions show Flutter's red error screen instead of a friendly recovery UI.

**Fix:** Add to `main.dart`:
```dart
FlutterError.onError = (details) {
  // log to crash service or show banner
};
```

---

### P3 — Biometrics screen is a dead stub
**File:** `lib/screens/features/phase5_screens.dart` — `BiometricInfoScreen`

Static text explaining that biometrics was removed. Appears as a real feature in the grid. Should be hidden or removed from the feature list.

---

### P4 — Admin screen not guarded on the server side
**File:** `lib/screens/admin/admin_screen.dart`

The screen is hidden in the UI based on `user.role == 'admin'`, but the underlying API calls (change user role, list all analyses) are not further protected client-side. If a non-admin guesses the route, they access admin data.

Note: Backend should enforce role — but the client should not surface admin actions to non-admins even via direct navigation.

---

## ✅ CONFIRMED WORKING

| Feature | Screen | API Endpoint |
|---|---|---|
| Login | `auth/login_screen.dart` | POST `/auth/login` |
| Register | `auth/register_screen.dart` | POST `/auth/register` |
| Email verification | `auth/verify_email_screen.dart` | POST `/auth/verify-email` |
| Forgot password (3-step) | `auth/forgot_password_screen.dart` | Full flow |
| Session guard + auto-logout | `auth_gate.dart` + `auth_provider.dart` | GET `/auth/me` |
| Home shell (5-tab nav) | `home/home_shell.dart` | — |
| Contract upload + analyze | `analyze/analyze_screen.dart` | POST `/ocr_check_and_search` |
| Analysis results | `history/analysis_detail_screen.dart` | — |
| Analysis history + delete | `history/history_screen.dart` | GET/DELETE `/analyses` |
| Chat (assistant) | `chat/chat_assistant_screen.dart` | POST `/chat/assistant` |
| Chat with document | `chat/chat_document_screen.dart` | POST `/chat/document` |
| Feed (infinite scroll, likes, comments) | `social/feed_screen.dart` | GET/POST `/api/posts` |
| Network (suggestions, invites, accept) | `social/network_screen.dart` | Full network API |
| Profile (self + edit) | `social/profile_screen.dart` | Profile API |
| Member profile | `social/member_profile_screen.dart` | — |
| Skills & endorsements | `social/profile_skills_screen.dart` | GET `/api/profile/{id}/endorsements` |
| Documents | `social/profile_documents_screen.dart` | Upload + list |
| Recommendations | `social/profile_recommendations_screen.dart` | GET `/api/profile/{id}/recommendations` |
| Alerts | `social/alerts_screen.dart` | Timeline + invites |
| Explain clause | `phase5_screens.dart` | POST `/legato/explain-clause` |
| Summarize clauses | `phase5_screens.dart` | POST `/legato/summarize-clauses` |
| Compare contracts | `phase5_screens.dart` | POST `/legato/compare` |
| Negotiation coach | `phase5_screens.dart` | POST `/legato/negotiation-chat` |
| Share analysis | `phase5_screens.dart` | POST `/legato/shares` |
| Deal messaging | `phase5_screens.dart` | Full CRUD `/legato/deal-threads` |
| Timeline (admin) | `phase5_screens.dart` | GET/POST `/legato/timeline` |
| Admin panel | `admin/admin_screen.dart` | Users + analyses |
| More screen / Swagger | `more/more_screen.dart` | — |
| Roadmap | `roadmap/roadmap_screen.dart` | — |

---

## 📱 REAL DEVICE SPECIFIC ISSUES

| # | Issue | Severity |
|---|---|---|
| R1 | API URL `10.0.2.2:8002` unreachable on real device — every API call fails | **BLOCKER** |
| R2 | No `READ_MEDIA_IMAGES` permission — file picker fails on Android 13+ | **BLOCKER** |
| R3 | `usesCleartextTraffic=true` is global — needed for HTTP dev server but a production liability | High |
| R4 | No `CAMERA` permission declared — if any future screen uses camera, it will crash | Medium |
| R5 | Long API timeouts (10 min for analyze, 5 min for chat) — phone screen lock will kill the request | Medium |
| R6 | No `WakeLock` during long analysis — screen dims and background thread may be killed by Android Doze | Medium |
| R7 | No loading state persisted across screen rotation — rotation during analysis/chat loses the result | Low |
| R8 | Font scale — large system font will overflow several tight containers in phase5 feature cards | Low |

---

## PRIORITY ORDER FOR FIXING

1. **C1** — Add Android 13 file permissions (15 min fix, manifest only)
2. **C2** — Set real API URL for device testing (build flag, no code change)
3. **C5** — Wire DashboardTab into HomeShell (30 min)
4. **M1** — Fix or hide feed search bar (30 min)
5. **C3** — Add session expiry dialog (1 hour)
6. **M2** — Parse risk JSON into visual display (2 hours)
7. **M4** — Add analysis picker to ChatAnalysisScreen (1 hour)
8. **M3** — Add signature canvas to E-sign (2-3 hours with `hand_signature` package)
9. **C4** — Scope cleartext traffic to dev domains only (1 hour)
10. **M5** — Fix share URL to use real public domain (30 min)

---

*End of audit. 42 Dart files scanned. 0 mock data found. All API calls are wired to real backend.*
