import 'dart:typed_data';

import 'package:legato_mobile/demo/demo_state.dart';
import 'package:legato_mobile/models/user_model.dart';
import 'package:legato_mobile/services/auth_service.dart';

/// Drop-in replacement for [AuthService] that never touches the network.
class DemoAuthService extends AuthService {
  DemoAuthService() : super();

  static bool _loggedOut = false;

  @override
  Future<bool> hasToken() async => !_loggedOut;

  @override
  Future<UserModel> me() async => DemoState.current;

  @override
  Future<Map<String, dynamic>> login(String email, String password) async {
    _loggedOut = false;
    DemoState.setUser(DemoState.forEmail(email));
    return {'access_token': 'demo_token'};
  }

  @override
  Future<Map<String, dynamic>> register(String email, String password,
      {String userType = 'user'}) async {
    _loggedOut = false;
    DemoState.setUser(DemoState.forEmail(email));
    return {'message': 'demo registration successful'};
  }

  /// Stores the uploaded CV and ID card in [DemoState] so the admin can
  /// later retrieve the real bytes when reviewing the application.
  @override
  Future<Map<String, dynamic>> registerAsLawyer({
    required String email,
    required String password,
    int? yearsOfExperience,
    Uint8List? cvBytes,
    String? cvFilename,
    Uint8List? idCardBytes,
    String? idCardFilename,
  }) async {
    _loggedOut = false;
    final user = DemoState.forEmail(email);
    DemoState.setUser(user);
    DemoState.submitLawyerApp(
      userId: user.id,
      cvFilename: cvFilename,
      cvBytes: cvBytes,
      idCardFilename: idCardFilename,
      idCardBytes: idCardBytes,
      yearsOfExperience: yearsOfExperience,
    );
    return {'message': 'demo lawyer registration successful'};
  }

  @override
  Future<Map<String, dynamic>> verifyEmail(
          {required String email, required String code}) async =>
      {'message': 'demo email verified'};

  @override
  Future<Map<String, dynamic>> resendVerification(String email) async =>
      {'message': 'demo resend ok'};

  @override
  Future<Map<String, dynamic>> forgotPassword(String email) async =>
      {'message': 'demo forgot password ok'};

  @override
  Future<void> logout() async => _loggedOut = true;

  @override
  Future<void> storeToken(String token) async {}
}
