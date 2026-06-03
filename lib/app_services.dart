import 'package:legato_mobile/api/api_client.dart';
import 'package:legato_mobile/api/legato_api.dart';
import 'package:legato_mobile/services/auth_service.dart';
import 'package:legato_mobile/storage/token_storage.dart';

/// Single shared [TokenStorage] + [ApiClient] so JWT applies to all calls.
class AppServices {
  AppServices() {
    storage = TokenStorage();
    api = ApiClient(
      storage: storage,
      onUnauthorized: () => _onUnauthorized?.call(),
    );
    auth = AuthService(api: api, storage: storage);
    legato = LegatoApi(api);
  }

  void Function()? _onUnauthorized;

  /// Called after the API clears the JWT (401). Keeps [AuthProvider] in sync with storage.
  void setOnUnauthorized(void Function()? cb) => _onUnauthorized = cb;

  late final TokenStorage storage;
  late final ApiClient api;
  late final AuthService auth;
  late final LegatoApi legato;
}
