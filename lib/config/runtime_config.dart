import 'package:shared_preferences/shared_preferences.dart';

import 'app_config.dart';

/// Mutable runtime counterpart to [AppConfig].
/// Reads the override URL from SharedPreferences at startup so demo operators
/// can switch between local and production without rebuilding the APK.
class RuntimeConfig {
  static const String _prefKey = 'api_base_url_override';

  /// Old HTTP VPS IP — browsers block this from Firebase HTTPS (mixed content).
  static const String _legacyHttpVps = 'http://76.13.4.148';

  static String _apiBaseUrl = AppConfig.apiBaseUrl;

  static String get apiBaseUrl => _apiBaseUrl;

  static String _normalize(String url) => url.replaceAll(RegExp(r'/$'), '');

  /// Upgrade stale overrides saved before HTTPS was enabled on the VPS.
  static String? _migrateLegacyOverride(String override) {
    final normalized = _normalize(override);
    if (normalized == _legacyHttpVps ||
        normalized.startsWith('http://76.13.4.148:') ||
        normalized.startsWith('http://76.13.4.148/')) {
      return _normalize(AppConfig.apiBaseUrl);
    }
    return normalized;
  }

  /// Call once in main() before runApp(). Safe to call multiple times.
  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    final override = prefs.getString(_prefKey);
    if (override != null && override.isNotEmpty) {
      final migrated = _migrateLegacyOverride(override);
      _apiBaseUrl = migrated ?? _normalize(override);
      if (migrated != null && migrated != _normalize(override)) {
        await prefs.setString(_prefKey, _apiBaseUrl);
      }
    }
  }

  /// Persist a new URL and apply it immediately (no app restart needed).
  static Future<void> setApiBaseUrl(String url) async {
    _apiBaseUrl = url.replaceAll(RegExp(r'/$'), '');
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKey, _apiBaseUrl);
  }

  /// Remove the override and fall back to the compile-time default.
  static Future<void> clearOverride() async {
    _apiBaseUrl = AppConfig.apiBaseUrl;
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_prefKey);
  }
}
