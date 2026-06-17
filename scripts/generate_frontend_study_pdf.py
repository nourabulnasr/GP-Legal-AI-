"""Generate Legato Flutter frontend study guide PDF for team discussions."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "Legato_Frontend_Study_Guide.pdf"


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontSize=22,
            spaceAfter=14,
            textColor=colors.HexColor("#1B1F23"),
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#666666"),
            spaceAfter=20,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=16,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#915907"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#1B1F23"),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            leftIndent=14,
            bulletIndent=0,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "CodeBlock",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=9.5,
            backColor=colors.HexColor("#F4F4F5"),
            borderPadding=6,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#444444"),
            spaceAfter=8,
            fontName="Helvetica-Bold",
        ),
    }


def P(text: str, style) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def code_block(text: str, style) -> Preformatted:
    return Preformatted(text.strip(), style)


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1B1F23")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def build_story():
    s = build_styles()
    story = []

    story.append(P("Legato Mobile — Flutter Frontend Study Guide", s["title"]))
    story.append(
        P(
            "GP-Legal-AI · legato_mobile · Package: lib/ · Live: legatoappgp2026.web.app<br/>"
            "Prepared for team discussion — block-by-block explanation of how the frontend works.",
            s["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#C7A006")))
    story.append(Spacer(1, 0.3 * cm))

    # --- Section 1 ---
    story.append(P("1. Big Picture — How the App Is Organized", s["h1"]))
    story.append(
        P(
            "The Flutter frontend talks to the FastAPI backend over HTTPS. "
            "State is managed with Provider. Navigation is simple: AuthGate decides login vs home; "
            "sub-screens use Navigator.push.",
            s["body"],
        )
    )
    story.append(
        code_block(
            """
main.dart → Provider (AppServices, AuthProvider, ThemeNotifier)
         → AuthGate
              ├─ logged in  → HomeShell (6 bottom tabs)
              └─ logged out → LoginScreen
         → Screens call LegatoApi → ApiClient → Backend + JWT in TokenStorage
""",
            s["code"],
        )
    )
    story.append(
        table(
            [
                ["Folder", "Role"],
                ["lib/main.dart", "App entry, providers, theme, lifecycle"],
                ["lib/config/", "API URL, timeouts, share links"],
                ["lib/storage/", "Save JWT token locally"],
                ["lib/api/", "HTTP client + all backend endpoints"],
                ["lib/services/", "Auth logic (login, register, OAuth)"],
                ["lib/providers/", "App-wide state (user, theme)"],
                ["lib/screens/", "Every page the user sees"],
                ["lib/widgets/", "Reusable UI components"],
                ["lib/theme/", "LinkedIn-style light/dark theme"],
                ["lib/utils/", "Platform helpers (PDF download, file bytes)"],
            ],
            col_widths=[4.5 * cm, 12 * cm],
        )
    )
    story.append(
        P(
            "<b>Pattern used everywhere:</b> A screen reads AppServices or AuthProvider via Provider, "
            "calls legato.someApiMethod(), then updates UI with setState or ChangeNotifier.",
            s["body"],
        )
    )

    story.append(PageBreak())

    # --- Section 2 ---
    story.append(P("2. App Startup — main.dart", s["h1"]))
    story.append(P("Block A — Initialize Flutter and config", s["h2"]))
    story.append(
        code_block(
            """
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RuntimeConfig.init();
  final themeNotifier = await ThemeNotifier.init();
  runApp(LegatoApp(themeNotifier: themeNotifier));
}
""",
            s["code"],
        )
    )
    story.append(P("What each line does:", s["caption"]))
    for line in [
        "ensureInitialized() — Required before async/plugin work in Flutter.",
        "RuntimeConfig.init() — Loads saved API URL override from device storage.",
        "ThemeNotifier.init() — Loads dark/light preference.",
        "runApp() — Starts the widget tree at LegatoApp.",
    ]:
        story.append(P(f"• {line}", s["bullet"]))

    story.append(P("Block B — Dependency injection with Provider", s["h2"]))
    story.append(
        code_block(
            """
MultiProvider(
  providers: [
    Provider<AppServices>(create: (_) => AppServices()),
    ChangeNotifierProvider<AuthProvider>(... bootstrap ...),
    ChangeNotifierProvider<ThemeNotifier>.value(value: themeNotifier),
  ],
  child: MaterialApp(home: AuthGate(), ...),
)
""",
            s["code"],
        )
    )
    story.append(
        table(
            [
                ["Provider", "Purpose"],
                ["AppServices", "One shared HTTP client + auth + API (JWT on every request)"],
                ["AuthProvider", "Current user, login/logout, session expiry"],
                ["ThemeNotifier", "Dark/light mode toggle"],
            ],
            col_widths=[4 * cm, 12.5 * cm],
        )
    )

    story.append(P("Block C — MaterialApp shell", s["h2"]))
    story.append(
        P(
            "• theme / darkTheme — LinkedIn-style gold/dark UI.<br/>"
            "• maxWidth: 600 — On web, app looks like a centered phone column.<br/>"
            "• home: AuthGate() — First screen decides login vs main app.",
            s["body"],
        )
    )
    story.append(
        P(
            "Block D — Lifecycle: when app resumes, AuthProvider.refreshUser() re-fetches /auth/me "
            "to detect expired JWT. On Android, deep links with ?token= trigger loginWithToken().",
            s["body"],
        )
    )

    story.append(PageBreak())

    # --- Section 3 ---
    story.append(P("3. Configuration Layer", s["h1"]))
    story.append(P("AppConfig — compile-time settings", s["h2"]))
    story.append(
        code_block(
            """
static const String apiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'https://srv1723974.hstgr.cloud',
);
""",
            s["code"],
        )
    )
    story.append(
        P(
            "Override at build: flutter build web --dart-define=API_BASE_URL=https://your-api.com<br/>"
            "Timeouts: 90s default, 10 min analyze, 5 min chat, 2 min auth.",
            s["body"],
        )
    )
    story.append(P("RuntimeConfig — runtime API URL override", s["h2"]))
    story.append(
        P(
            "Settings screen can change API URL without rebuild. Stored in SharedPreferences. "
            "Migrates old http://76.13.4.148 to HTTPS (browsers block mixed content from Firebase).",
            s["body"],
        )
    )

    story.append(P("4. Storage — JWT Token (TokenStorage)", s["h1"]))
    story.append(
        code_block(
            """
Key: 'access_token' in SharedPreferences
readToken()  → attach to Authorization: Bearer ...
writeToken() → after login / OAuth
clearToken() → on 401 or logout
""",
            s["code"],
        )
    )

    story.append(P("5. HTTP Layer — ApiClient", s["h1"]))
    story.append(P("Build URL + auth header", s["h2"]))
    story.append(
        code_block(
            """
Uri uri(String path) => Uri.parse('${RuntimeConfig.apiBaseUrl}$path');

_headers():
  Content-Type: application/json (for POST)
  Authorization: Bearer <token>  (if logged in)
""",
            s["code"],
        )
    )
    story.append(P("Handle 401 (session expired)", s["h2"]))
    story.append(
        P(
            "If statusCode == 401 → clearToken() → call onUnauthorized → AuthProvider shows "
            "'Session expired' dialog.",
            s["body"],
        )
    )
    story.append(P("Typical JSON POST pattern", s["h2"]))
    story.append(
        P(
            "postJson(path, body) → POST with JSON → parse response OR throw ApiException with server message. "
            "Screens catch ApiException and show red error text.",
            s["body"],
        )
    )
    story.append(
        P(
            "Also: postMultipartOcrCheck for PDF/DOCX upload to /ocr_check_and_search (long timeout).",
            s["body"],
        )
    )

    story.append(PageBreak())

    # --- Section 6-8 ---
    story.append(P("6. AppServices — Service Wiring", s["h1"]))
    story.append(
        code_block(
            """
AppServices() {
  storage = TokenStorage();
  api = ApiClient(storage: storage, onUnauthorized: ...);
  auth = AuthService(api: api, storage: storage);
  legato = LegatoApi(api);
}
""",
            s["code"],
        )
    )
    story.append(P("One chain: TokenStorage → ApiClient → AuthService + LegatoApi. All screens share one token.", s["body"]))

    story.append(P("7. AuthService — Login / Register / OAuth", s["h1"]))
    story.append(
        table(
            [
                ["Method", "Backend route", "Purpose"],
                ["login", "POST /auth/login", "Email/password → save JWT"],
                ["register", "POST /auth/register", "Create account"],
                ["verifyEmail", "POST /auth/verify-email", "Confirm email code"],
                ["resendVerification", "POST /auth/resend-verification", "Resend code (with retry)"],
                ["loginWithGoogleIdToken", "POST /auth/google/id-token", "Native Google"],
                ["exchangeGoogleCode", "POST /auth/google/code", "Web OAuth redirect"],
                ["me", "GET /auth/me", "Current user profile"],
            ],
            col_widths=[4.5 * cm, 5.5 * cm, 6.5 * cm],
        )
    )

    story.append(P("8. AuthProvider — Auth State", s["h1"]))
    story.append(P("bootstrap() on app start:", s["h2"]))
    for line in [
        "Web OAuth ?code= → exchange for JWT via exchangeGoogleCode()",
        "Deep link ?token= → storeToken() + me()",
        "Saved token exists → GET /auth/me",
        "Sets _user or null → drives login vs home",
    ]:
        story.append(P(f"• {line}", s["bullet"]))
    story.append(
        P(
            "Exposed: user, loading, isAuthenticated. Screens use context.watch&lt;AuthProvider&gt;() to rebuild on login/logout.",
            s["body"],
        )
    )

    story.append(PageBreak())

    # --- Section 9-11 ---
    story.append(P("9. AuthGate — Navigation Gate", s["h1"]))
    story.append(
        code_block(
            """
if (auth.loading)     → spinner
if (session expired)   → dialog "Session expired"
if (isAuthenticated)   → HomeShell
else                  → LoginScreen
""",
            s["code"],
        )
    )
    story.append(P("No router package — simple if/else on home: + Navigator.push for sub-pages.", s["body"]))

    story.append(P("10. HomeShell — Bottom Navigation", s["h1"]))
    story.append(
        table(
            [
                ["Tab", "Screen", "Purpose"],
                ["0 Home", "DashboardTab", "Quick links, stats"],
                ["1 Feed", "FeedScreen", "Social posts"],
                ["2 Network", "NetworkScreen", "Connections, invites"],
                ["3 Contracts", "ContractsTabScreen", "User analyses list"],
                ["4 Alerts", "AlertsScreen", "Notifications"],
                ["5 Profile", "ProfileScreen", "Own profile edit"],
            ],
            col_widths=[2.5 * cm, 4.5 * cm, 9.5 * cm],
        )
    )
    story.append(
        P(
            "IndexedStack keeps all tabs in memory — switching tabs is instant; scroll position is preserved.",
            s["body"],
        )
    )

    story.append(P("11. LegatoApi — API Facade", s["h1"]))
    story.append(
        table(
            [
                ["Group", "Example methods", "Backend"],
                ["Analysis", "analyzeContract, listAnalyses", "/ocr_check_and_search, /analyses"],
                ["Chat", "chatMessage, chatAssistant", "/chat/*"],
                ["Social", "getFeed, createPost, sendNetworkInvite", "/api/posts, /api/network/*"],
                ["Profile", "getSocialProfile, putApiProfile", "/api/profile/*"],
                ["Admin", "adminListUsers, adminDeleteUser", "/analyses/admin/*"],
            ],
            col_widths=[2.5 * cm, 5.5 * cm, 8.5 * cm],
        )
    )
    story.append(
        P(
            "Resilient helpers: getSocialProfileResilient tries /api/profile/{id}, falls back to /legato/profile/me shape if 404.",
            s["body"],
        )
    )

    story.append(PageBreak())

    # --- Section 12-13 ---
    story.append(P("12. Standard Screen Pattern", s["h1"]))
    story.append(
        code_block(
            """
class _SomeScreenState extends State<SomeScreen> {
  bool _loading = true;
  String? _err;
  List<dynamic> _data = [];

  void initState() { super.initState(); _load(); }

  Future<void> _load() async {
    setState(() { _loading = true; _err = null; });
    try {
      final data = await context.read<AppServices>().legato.someMethod();
      setState(() { _data = data; _loading = false; });
    } on ApiException catch (e) {
      setState(() { _err = e.message; _loading = false; });
    }
  }

  Widget build() {
    if (_loading) return CircularProgressIndicator();
    if (_err != null) return Text(_err!);
    return ListView(...);
  }
}
""",
            s["code"],
        )
    )

    story.append(P("13. Feature Screens Summary", s["h1"]))
    story.append(P("Auth (lib/screens/auth/)", s["h2"]))
    story.append(
        P(
            "login_screen — email/password; Google OAuth (web redirect vs mobile browser).<br/>"
            "register_screen — sign up → verify email.<br/>"
            "verify_email_screen — enter code; resend code.<br/>"
            "forgot_password_screen — reset password flow.<br/>"
            "Google on web: OAuth → redirect to Firebase with ?code= → bootstrap exchanges for JWT.",
            s["body"],
        )
    )
    story.append(P("Analyze (analyze_screen.dart)", s["h2"]))
    story.append(
        P(
            "FilePicker → analyzeContract(bytes) → backend OCR + RAG + ML + LLM → optional save → AnalysisDetailScreen. "
            "WakelockPlus prevents sleep during long analysis.",
            s["body"],
        )
    )
    story.append(P("Social (lib/screens/social/)", s["h2"]))
    story.append(
        P(
            "feed_screen — posts, like/comment/share (WhatsApp uses AppConfig.shareBaseUrl).<br/>"
            "network_screen — suggestions, invites, connections.<br/>"
            "member_profile_screen — view user; Connect/Connected via connection_status.<br/>"
            "profile_screen — edit own profile, avatar, skills.",
            s["body"],
        )
    )
    story.append(P("Chat, Admin, More", s["h2"]))
    story.append(
        P(
            "chat_assistant_screen — POST /chat/assistant.<br/>"
            "chat_analysis_screen — chat about saved analysis.<br/>"
            "admin_screen — list users, change roles, delete users (admin only).<br/>"
            "more_screen — links to Features hub, Settings, Roadmap, Admin.",
            s["body"],
        )
    )

    story.append(PageBreak())

    # --- Section 14-20 ---
    story.append(P("14. Reusable Widgets", s["h1"]))
    story.append(
        table(
            [
                ["Widget", "File", "Role"],
                ["LegatoAppBar", "legato_app_bar.dart", "Top bar with back button"],
                ["UserAvatar", "user_avatar.dart", "Circle avatar from URL or initials"],
                ["AnalysisIdPicker", "analysis_id_picker.dart", "Dropdown of user analyses"],
                ["DocumentChatBubble", "document_chat_bubble.dart", "Chat message UI"],
            ],
            col_widths=[3.5 * cm, 5 * cm, 8 * cm],
        )
    )

    story.append(P("15. Theme (linkedin_theme.dart)", s["h1"]))
    story.append(
        P(
            "Gold accent (navActiveGold), dark scaffold colors, card styles. "
            "textSecondaryAdaptive(context) works in light and dark mode.",
            s["body"],
        )
    )

    story.append(P("16. Platform Utils (lib/utils/)", s["h1"]))
    story.append(
        P(
            "Conditional imports for web vs mobile:<br/>"
            "text_download.dart → web or io implementation<br/>"
            "pdf_download.dart — same pattern<br/>"
            "platform_file_bytes.dart — gallery/camera bytes on web vs native",
            s["body"],
        )
    )

    story.append(P("17. End-to-End: Admin Deletes User", s["h1"]))
    story.append(
        code_block(
            """
AdminScreen._deleteUser()
  → showDialog confirm
  → legato.adminDeleteUser(userId)
    → ApiClient.deleteJson('/analyses/admin/users/$id')
  → _load() refreshes list
  → SnackBar "Deleted email@..."
""",
            s["code"],
        )
    )

    story.append(P("18. End-to-End: View Friend Profile", s["h1"]))
    story.append(
        code_block(
            """
MemberProfileScreen._load()
  → getSocialProfileResilient(userId)
  → if no connection_status, check getNetworkConnections()
  → UI shows "Connected" instead of "Connect"
""",
            s["code"],
        )
    )

    story.append(P("19. How to Read Any Screen Quickly", s["h1"]))
    for i, line in enumerate(
        [
            "Find initState / _load() — what API is called?",
            "Find context.read<AppServices>() — which legato.* method?",
            "Find build() — loading / error / success UI",
            "Find Navigator.push — where can user go next?",
        ],
        1,
    ):
        story.append(P(f"{i}. {line}", s["bullet"]))

    story.append(P("20. Discussion Talking Points", s["h1"]))
    points = [
        "Why Provider instead of Riverpod/Bloc? — Simple shared services + auth state; matches small team scope.",
        "Why IndexedStack for tabs? — Preserve state; faster tab switching.",
        "How is JWT security handled? — Stored in SharedPreferences; cleared on 401; not in URL except OAuth handoff.",
        "Web vs mobile differences? — Google OAuth redirect on web; deep links on Android; maxWidth 600 on web.",
        "How does the app survive backend changes? — Resilient API helpers (getSocialProfileResilient).",
        "Where is the API URL configured? — AppConfig compile-time + RuntimeConfig runtime override in Settings.",
        "What happens when analysis takes 5+ minutes? — longTimeout (10 min), WakelockPlus, progress status text.",
    ]
    for p in points:
        story.append(P(f"• {p}", s["bullet"]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    story.append(
        P(
            "<i>Generated from GP-Legal-AI Flutter frontend (legato_mobile). "
            "Repo: lib/ · 61 Dart files · Live: legatoappgp2026.web.app</i>",
            s["subtitle"],
        )
    )

    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Legato Flutter Frontend Study Guide",
        author="GP-Legal-AI Team",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(2 * cm, 1.2 * cm, "Legato Frontend Study Guide — GP-Legal-AI")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
