import 'package:flutter/foundation.dart';
import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/models/user_model.dart';
import 'package:legato_mobile/services/auth_service.dart';

class AuthProvider extends ChangeNotifier {
  AuthProvider({AuthService? authService}) : _auth = authService ?? AuthService();

  final AuthService _auth;

  UserModel? _user;
  bool _loading = true;
  String? _error;

  UserModel? get user => _user;
  bool get loading => _loading;
  String? get error => _error;
  bool get isAuthenticated => _user != null;

  Future<void> bootstrap() async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      final has = await _auth.hasToken();
      if (!has) {
        _user = null;
        return;
      }
      _user = await _auth.me();
    } on ApiException catch (e) {
      _user = null;
      _error = e.message;
    } catch (e) {
      _user = null;
      _error = e.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> login(String email, String password) async {
    _error = null;
    await _auth.login(email, password);
    _user = await _auth.me();
    notifyListeners();
  }

  Future<void> register(String email, String password) async {
    _error = null;
    await _auth.register(email, password);
    notifyListeners();
  }

  Future<void> logout() async {
    await _auth.logout();
    _user = null;
    notifyListeners();
  }

  /// JWT was rejected (401); storage is already cleared by [ApiClient].
  void onSessionExpiredFromApi() {
    _user = null;
    _error = 'Session expired. Please sign in again.';
    notifyListeners();
  }

  Future<void> refreshUser() async {
    if (!await _auth.hasToken()) {
      if (_user != null) {
        _user = null;
        notifyListeners();
      }
      return;
    }
    try {
      _user = await _auth.me();
      _error = null;
      notifyListeners();
    } on ApiException catch (e) {
      if (e.statusCode == 401) {
        _user = null;
        _error = 'Session expired. Please sign in again.';
        notifyListeners();
      }
    } catch (_) {}
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }
}
