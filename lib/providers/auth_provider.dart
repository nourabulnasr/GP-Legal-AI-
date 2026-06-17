import 'package:flutter/foundation.dart';
import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/config/runtime_config.dart';
import 'package:legato_mobile/models/user_model.dart';
import 'package:legato_mobile/services/auth_service.dart';
import 'package:legato_mobile/utils/local_lawyer_queue.dart';
import 'package:legato_mobile/utils/pending_lawyer_docs.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

class AuthProvider extends ChangeNotifier {
  AuthProvider({AuthService? authService}) : _auth = authService ?? AuthService();

  final AuthService _auth;

  UserModel? _user;
  bool _loading = true;
  String? _error;

  // ── Local lawyer-type override ────────────────────────────────────────────
  // The remote server may be running old code that doesn't store/return
  // user_type='lawyer'. We keep a local set of emails that registered as
  // lawyers so the AuthGate can route them correctly regardless of the server.
  static const _lawyerEmailsKey = 'lawyer_registered_emails';

  Future<void> _storeLawyerEmail(String email) async {
    final p = await SharedPreferences.getInstance();
    final list = p.getStringList(_lawyerEmailsKey) ?? [];
    final norm = email.trim().toLowerCase();
    if (!list.contains(norm)) {
      list.add(norm);
      await p.setStringList(_lawyerEmailsKey, list);
    }
  }

  Future<bool> _isStoredLawyer(String email) async {
    final p = await SharedPreferences.getInstance();
    final list = p.getStringList(_lawyerEmailsKey) ?? [];
    return list.contains(email.trim().toLowerCase());
  }

  /// Ensures the cached UserModel reflects any lawyer status that the server
  /// may not know about yet (local queue submission / local admin approval).
  ///
  /// Priority: backend data > local queue status > stored-email override.
  Future<UserModel> _withLawyerOverride(UserModel user) async {
    // Fast path: backend already reports fully verified.
    if (user.isVerifiedLawyer) return user;

    // Determine if this account is a lawyer by backend or local email set.
    final isLocalLawyer = user.userType == 'lawyer' || await _isStoredLawyer(user.email);

    // Check the local queue for the most recent status (e.g. admin approved
    // locally when backend was unavailable).
    String? queueStatus;
    try {
      final all = await LocalLawyerQueue.getAll();
      for (final entry in all) {
        if ((entry['user_id'] as int?) == user.id) {
          queueStatus = entry['status']?.toString();
          break;
        }
      }
    } catch (_) {}

    if (!isLocalLawyer && queueStatus == null) return user;

    final effectiveUserType = (isLocalLawyer || queueStatus != null) ? 'lawyer' : user.userType;
    final effectiveStatus   = queueStatus ?? user.lawyerStatus ?? 'not_applied';

    return UserModel(
      id: user.id,
      email: user.email,
      role: user.role,
      userType: effectiveUserType,
      lawyerStatus: effectiveStatus,
    );
  }
  // ─────────────────────────────────────────────────────────────────────────

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
        _user = await _withLawyerOverride(await _auth.me());
        return;
      }
      if (uriCode != null && uriCode.isNotEmpty) {
        final redirectUri = Uri.base.origin;
        // state carries user_type when coming from the register screen Google flow
        final stateUserType = Uri.base.queryParameters['state'] ?? 'user';
        final userType = (stateUserType == 'lawyer') ? 'lawyer' : 'user';
        await _auth.exchangeGoogleCode(uriCode, redirectUri: redirectUri, userType: userType);
        final rawUser = await _auth.me();
        if (userType == 'lawyer') await _storeLawyerEmail(rawUser.email);
        _user = await _withLawyerOverride(rawUser);
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
      _user = await _withLawyerOverride(await _auth.me());
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
      _user = await _withLawyerOverride(await _auth.me());
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
      final rawUser = await _auth.me();
      if (userType == 'lawyer') await _storeLawyerEmail(rawUser.email);
      _user = await _withLawyerOverride(rawUser);
    } on ApiException catch (e) {
      _error = e.message;
    } catch (e) {
      _error = e.toString();
    }
    notifyListeners();
  }

  /// Redirect the browser to Google sign-in (web only).
  /// Uses the current page origin as redirect_uri so both the auth request and
  /// the code exchange in bootstrap() send the same URI — required by Google.
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
    _user = await _withLawyerOverride(await _auth.me());
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
      try {
        await _auth.registerAsLawyer(
          email: email,
          password: password,
          yearsOfExperience: yearsOfExperience,
          cvBytes: cvBytes,
          cvFilename: cvFilename,
          idCardBytes: idCardBytes,
          idCardFilename: idCardFilename,
        );
      } on ApiException catch (e) {
        // Old server doesn't have /auth/register-lawyer yet — fall back to
        // regular JSON registration and stash docs for LawyerApplicationScreen.
        if (e.statusCode == 404 || e.statusCode == 405 || e.statusCode == 422) {
          await _auth.register(email, password, userType: userType);
          PendingLawyerDocs.cvBytes = cvBytes;
          PendingLawyerDocs.cvFilename = cvFilename;
          PendingLawyerDocs.idCardBytes = idCardBytes;
          PendingLawyerDocs.idCardFilename = idCardFilename;
        } else {
          rethrow;
        }
      }
    } else {
      await _auth.register(email, password, userType: userType);
    }
    if (userType == 'lawyer') await _storeLawyerEmail(email);
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
      _user = await _withLawyerOverride(await _auth.me());
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
