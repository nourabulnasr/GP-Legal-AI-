import 'dart:typed_data';

import 'package:legato_mobile/api/api_client.dart';
import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/config/app_config.dart';
import 'package:legato_mobile/models/user_model.dart';
import 'package:legato_mobile/storage/token_storage.dart';

/// Mirrors `legalai-frontend/src/lib/auth.ts`.
class AuthService {
  AuthService({ApiClient? api, TokenStorage? storage})
      : _api = api ?? ApiClient(),
        _storage = storage ?? TokenStorage();

  final ApiClient _api;
  final TokenStorage _storage;

  Future<Map<String, dynamic>> login(String email, String password) async {
    final data = await _postJsonWithRetry(
      '/auth/login',
      {
        'email': email.trim(),
        'password': password,
      },
      timeout: AppConfig.authTimeout,
    );
    final token = data['access_token'] as String?;
    if (token == null || token.isEmpty) {
      throw ApiException('No access token returned');
    }
    await _storage.writeToken(token);
    return data;
  }

  /// Retries transient network failures (e.g. connection abort while backend is cold).
  Future<Map<String, dynamic>> _postJsonWithRetry(
    String path,
    Map<String, dynamic> body, {
    required Duration timeout,
    int maxAttempts = 3,
  }) async {
    Object? lastError;
    for (var attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await _api.postJson(path, body, timeout: timeout);
      } catch (e, st) {
        lastError = e;
        final retryable = _isRetryableNetworkError(e);
        if (!retryable || attempt == maxAttempts) {
          Error.throwWithStackTrace(e, st);
        }
        await Future<void>.delayed(Duration(milliseconds: 400 * attempt));
      }
    }
    throw lastError ?? ApiException('Login request failed');
  }

  bool _isRetryableNetworkError(Object e) {
    final s = e.toString().toLowerCase();
    return s.contains('socketexception') ||
        s.contains('connection abort') ||
        s.contains('connection reset') ||
        s.contains('connection refused') ||
        s.contains('failed host lookup') ||
        s.contains('timed out');
  }

  Future<Map<String, dynamic>> loginWithGoogleIdToken(String idToken, {String userType = 'user'}) async {
    final data = await _postJsonWithRetry(
      '/auth/google/id-token',
      {'id_token': idToken, 'user_type': userType},
      timeout: AppConfig.authTimeout,
    );
    final token = data['access_token'] as String?;
    if (token == null || token.isEmpty) {
      throw ApiException('No access token returned');
    }
    await _storage.writeToken(token);
    return data;
  }

  Future<Map<String, dynamic>> exchangeGoogleCode(String code, {required String redirectUri, String userType = 'user'}) async {
    final data = await _postJsonWithRetry(
      '/auth/google/code',
      {
        'code': code,
        'redirect_uri': redirectUri,
        'user_type': userType,
      },
      timeout: AppConfig.authTimeout,
    );
    final token = data['access_token'] as String?;
    if (token == null || token.isEmpty) {
      throw ApiException('No access token returned');
    }
    await _storage.writeToken(token);
    return data;
  }

  Future<Map<String, dynamic>> googleSignInConfig() => _api.getJson('/auth/google/config');

  Future<Map<String, dynamic>> register(String email, String password, {String userType = 'user'}) {
    return _postJsonWithRetry(
      '/auth/register',
      {
        'email': email.trim(),
        'password': password,
        'user_type': userType,
      },
      timeout: AppConfig.authTimeout,
    );
  }

  Future<Map<String, dynamic>> registerAsLawyer({
    required String email,
    required String password,
    int? yearsOfExperience,
    double? hourlyRate,
    Uint8List? cvBytes,
    String? cvFilename,
    Uint8List? idCardBytes,
    String? idCardFilename,
    Uint8List? idCardBackBytes,
    String? idCardBackFilename,
  }) {
    return _api.postMultipartRegisterLawyer(
      email: email,
      password: password,
      yearsOfExperience: yearsOfExperience,
      hourlyRate: hourlyRate,
      cvBytes: cvBytes,
      cvFilename: cvFilename,
      idCardBytes: idCardBytes,
      idCardFilename: idCardFilename,
      idCardBackBytes: idCardBackBytes,
      idCardBackFilename: idCardBackFilename,
    );
  }

  Future<UserModel> me() async {
    final data = await _api.getJson('/auth/me');
    return UserModel.fromJson(data);
  }

  Future<void> storeToken(String token) => _storage.writeToken(token);

  Future<void> logout() => _storage.clearToken();

  Future<Map<String, dynamic>> verifyEmail({required String email, required String code}) {
    return _api.postJson('/auth/verify-email', {
      'email': email.trim(),
      'code': code.trim(),
    });
  }

  Future<Map<String, dynamic>> resendVerification(String email) {
    return _postJsonWithRetry(
      '/auth/resend-verification',
      {'email': email.trim().toLowerCase()},
      timeout: AppConfig.authTimeout,
    );
  }

  Future<bool> hasToken() async {
    final t = await _storage.readToken();
    return t != null && t.isNotEmpty;
  }

  Future<Map<String, dynamic>> forgotPassword(String email) {
    return _api.postJson('/auth/forgot-password', {'email': email.trim()});
  }

  Future<Map<String, dynamic>> verifyResetCode({
    required String email,
    required String code,
  }) {
    return _api.postJson('/auth/verify-reset-code', {
      'email': email.trim(),
      'code': code.trim(),
    });
  }

  Future<Map<String, dynamic>> resetPassword({
    required String resetToken,
    required String newPassword,
  }) {
    return _api.postJson('/auth/reset-password', {
      'token': resetToken,
      'new_password': newPassword,
    });
  }
}
