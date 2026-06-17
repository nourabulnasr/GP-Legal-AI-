import 'package:flutter/foundation.dart';
import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/config/runtime_config.dart';
import 'package:legato_mobile/models/user_model.dart';
import 'package:legato_mobile/services/auth_service.dart';
import 'package:url_launcher/url_launcher.dart';

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
        // state carries user_type when coming from the register screen Google flow
        final stateUserType = Uri.base.queryParameters['state'] ?? 'user';
        final userType = (stateUserType == 'lawyer') ? 'lawyer' : 'user';
        await _auth.exchangeGoogleCode(uriCode, redirectUri: redirectUri, userType: userType);
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

  Future<void> loginWithGoogleIdToken(String idToken, {String userType = 'user'}) async {
    _error = null;
    try {
      await _auth.loginWithGoogleIdToken(idToken, userType: userType);
      _user = await _auth.me();
    } on ApiException catch (e) {
      _error = e.message;
    } catch (e) {
      _error = e.toString();
    }
    notifyListeners();
  }

  /// Web: redirect the browser to Google sign-in.
  /// Uses the current page origin as redirect_uri so both the auth request and
  /// the code exchange in [bootstrap] send the same URI — required by Google.
  /// Register your app's origin in Google Cloud Console → Authorized redirect URIs.
  Future<String?> signInWithGoogleWeb({String userType = 'user'}) async {
    _error = null;
    notifyListeners();
    try {
      final cfg = await _auth.googleSignInConfig();
      final clientId = cfg['client_id']?.toString().trim() ?? '';
      if (clientId.isEmpty || cfg['enabled'] != true) {
        _error = 'Google sign-in is not configured on the server.';
        notifyListeners();
        return _error;
      }
      final redirectUri = Uri.base.origin;
      final authUrl = Uri.https('accounts.google.com', '/o/oauth2/v2/auth', {
        'client_id': clientId,
        'redirect_uri': redirectUri,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'online',
        'prompt': 'select_account',
        'state': userType,
      });
      await launchUrl(authUrl, webOnlyWindowName: '_self');
      return null;
    } on ApiException catch (e) {
      _error = e.message;
      notifyListeners();
      return _error;
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      return _error;
    }
  }

  /// Mobile/desktop: launch backend Google OAuth redirect with user_type.
  Future<void> signInWithGoogleNative({String userType = 'user'}) async {
    final uri = Uri.parse('${RuntimeConfig.apiBaseUrl}/auth/google?user_type=$userType');
    if (!await canLaunchUrl(uri)) return;
    await launchUrl(uri, mode: LaunchMode.externalApplication);
  }

  Future<void> login(String email, String password) async {
    _error = null;
    await _auth.login(email, password);
    _user = await _auth.me();
    notifyListeners();
  }

  Future<void> register(
    String email,
    String password, {
    String userType = 'user',
    int? yearsOfExperience,
    Uint8List? cvBytes,
    String? cvFilename,
    Uint8List? idCardBytes,
    String? idCardFilename,
  }) async {
    _error = null;
    if (userType == 'lawyer' && (cvBytes != null || idCardBytes != null)) {
      await _auth.registerAsLawyer(
        email: email,
        password: password,
        yearsOfExperience: yearsOfExperience,
        cvBytes: cvBytes,
        cvFilename: cvFilename,
        idCardBytes: idCardBytes,
        idCardFilename: idCardFilename,
      );
    } else {
      await _auth.register(email, password, userType: userType);
    }
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
