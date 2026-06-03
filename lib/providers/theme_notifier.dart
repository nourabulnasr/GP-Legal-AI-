import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ThemeNotifier extends ChangeNotifier {
  static const _key = 'legato_theme_dark';

  ThemeNotifier._(this._isDark);

  bool _isDark;
  bool get isDark => _isDark;

  static Future<ThemeNotifier> init() async {
    final prefs = await SharedPreferences.getInstance();
    return ThemeNotifier._(prefs.getBool(_key) ?? false);
  }

  Future<void> toggle() async {
    _isDark = !_isDark;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_key, _isDark);
  }
}
