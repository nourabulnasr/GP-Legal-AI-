import 'package:shared_preferences/shared_preferences.dart';

/// Same key as web `localStorage` — mirrors legalai-frontend `api.ts`.
///
/// Uses [SharedPreferences] instead of flutter_secure_storage so Windows builds
/// do not pull `jni` → `objective_c` (those hooks break when `PUB_CACHE` or the
/// SDK path contains spaces, e.g. `C:\Users\Aly ahmed\...`).
class TokenStorage {
  static const _key = 'access_token';

  Future<String?> readToken() async {
    final p = await SharedPreferences.getInstance();
    return p.getString(_key);
  }

  Future<void> writeToken(String token) async {
    final p = await SharedPreferences.getInstance();
    await p.setString(_key, token);
  }

  Future<void> clearToken() async {
    final p = await SharedPreferences.getInstance();
    await p.remove(_key);
  }
}
