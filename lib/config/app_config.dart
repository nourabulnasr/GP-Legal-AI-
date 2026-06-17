class AppConfig {
  /// Production API on Hostinger VPS (HTTPS). Override at build: --dart-define=API_BASE_URL=...
  /// Local dev: http://10.0.2.2:8000 (emulator) or http://<LAN_IP>:8000 (physical device).
  static const String apiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'https://srv1723974.hstgr.cloud',
  );

  /// Optional override for web admin / Swagger (e.g. separate Next.js host). If empty, uses [backendDocsUri].
  static const String webAdminBaseUrl = String.fromEnvironment('WEB_ADMIN_URL', defaultValue: '');

  /// Base URL used when constructing public share links (web app, not API).
  /// Override at build time: --dart-define=SHARE_BASE_URL=https://yourdomain.com
  static const String _shareBaseRaw = String.fromEnvironment('SHARE_BASE_URL', defaultValue: '');
  static const String _defaultShareBaseUrl = 'https://legatoappgp2026.web.app';
  static String get shareBaseUrl {
    if (_shareBaseRaw.isNotEmpty) {
      return _shareBaseRaw.replaceAll(RegExp(r'/$'), '');
    }
    return _defaultShareBaseUrl;
  }

  /// OpenAPI docs on the same host as the API (`/docs`); includes admin-law routes in the schema.
  static Uri backendDocsUri() {
    if (webAdminBaseUrl.isNotEmpty) {
      return Uri.parse(webAdminBaseUrl);
    }
    return Uri.parse(apiBaseUrl).resolve('docs');
  }

  /// Mirrors web `api.ts` timeout for analyze (600000 ms).
  static const Duration longTimeout = Duration(minutes: 10);

  /// Document chat can be long (web uses 300000 ms).
  static const Duration chatTimeout = Duration(minutes: 5);

  /// Default HTTP deadline for JSON calls (cold backend + DB can be slow).
  static const Duration defaultTimeout = Duration(seconds: 90);

  /// Login/register — allow extra time when the server is still loading models.
  static const Duration authTimeout = Duration(seconds: 120);
}