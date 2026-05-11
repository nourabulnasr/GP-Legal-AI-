import 'package:shared_preferences/shared_preferences.dart';

import 'app_config.dart';

/// Mutable runtime counterpart to [AppConfig].
/// Reads the override URL from SharedPreferences at startup so demo operators
/// can switch between local and production without rebuilding the APK.
class RuntimeConfig {
  static const String _prefKey = 'api_base_url_override';

  static String _apiBaseUrl = AppConfig.apiBaseUrl;

  static String get apiBaseUrl => _apiBaseUrl;

  /// Call once in main() before runApp(). Safe to call multiple times.
  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    final override = prefs.getString(_prefKey);
    if (override != null && override.isNotEmpty) {
      _apiBaseUrl = override.replaceAll(RegExp(r'/$'), '');
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
