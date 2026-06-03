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
      // Flutter web: ?token= from backend redirect, or ?code= from Firebase OAuth redirect.
      final uriToken = Uri.base.queryParameters['token'];
      final uriError = Uri.base.queryParameters['error'];
      final uriCode = Uri.base.queryParameters['code'];
      if (uriToken != null && uriToken.isNotEmpty) {
        await _auth.storeToken(uriToken);
        _user = await _auth.me();
        return;
      }
      if (uriCode != null && uriCode.isNotEmpty) {
        final redirectUri = Uri.base.origin;
        await _auth.exchangeGoogleCode(uriCode, redirectUri: redirectUri);
        _user = await _auth.me();
        return;
      }
      if (uriError != null && uriError.isNotEmpty) {
        _error = _oauthErrorMessage(uriError);
        _user = null;
        return;
      }
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

  Future<void> loginWithToken(String token) async {
    _error = null;
    try {
      await _auth.storeToken(token);
      _user = await _auth.me();
    } on ApiException catch (e) {
      _error = e.message;
    } catch (e) {
      _error = e.toString();
    }
    notifyListeners();
  }

  String _oauthErrorMessage(String code) => switch (code) {
        'google_denied' => 'Google sign-in was cancelled.',
        'google_not_configured' => 'Google sign-in is not set up on the server.',
        'google_token_failed' => 'Google authentication failed. Please try again.',
        'google_user_failed' => 'Could not retrieve your Google account details.',
        'google_no_email' => 'Your Google account has no email address.',
        _ => 'Google sign-in failed. Please try again.',
      };

  Future<void> loginWithGoogleIdToken(String idToken) async {
    _error = null;
    try {
      await _auth.loginWithGoogleIdToken(idToken);
      _user = await _auth.me();
    } on ApiException catch (e) {
      _error = e.message;
    } catch (e) {
      _error = e.toString();
    }
    notifyListeners();
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
