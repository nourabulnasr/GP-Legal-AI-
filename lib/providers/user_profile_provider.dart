import 'package:flutter/foundation.dart';

import 'package:legato_mobile/api/legato_api.dart';

/// Current user's social profile fields used across tabs (avatar, display name).
class UserProfileProvider extends ChangeNotifier {
  UserProfileProvider(this._legato);

  final LegatoApi _legato;

  String? _avatarUrl;
  int _avatarVersion = 0;
  String _displayName = '';

  String? get avatarUrl {
    final raw = _avatarUrl?.trim();
    if (raw == null || raw.isEmpty) return null;
    final sep = raw.contains('?') ? '&' : '?';
    return '$raw${sep}v=$_avatarVersion';
  }

  String get displayName => _displayName;

  /// Prefer live avatar for the signed-in user; otherwise use the stored URL.
  String? avatarForUser(int? userId, int? currentUserId, String? fallbackUrl) {
    if (userId != null && currentUserId != null && userId == currentUserId) {
      return avatarUrl ?? fallbackUrl;
    }
    return fallbackUrl;
  }

  void applyFromProfile(Map<String, dynamic> data, {String fallbackName = ''}) {
    final url = data['avatar_url']?.toString().trim();
    final name = data['display_name']?.toString().trim();
    _avatarUrl = (url != null && url.isNotEmpty) ? url : _avatarUrl;
    if (name != null && name.isNotEmpty) {
      _displayName = name;
    } else if (_displayName.isEmpty && fallbackName.isNotEmpty) {
      _displayName = fallbackName;
    }
    notifyListeners();
  }

  void setAvatarUrl(String url) {
    final trimmed = url.trim();
    if (trimmed.isEmpty) return;
    _avatarUrl = trimmed;
    _avatarVersion = DateTime.now().millisecondsSinceEpoch;
    notifyListeners();
  }

  Future<void> refresh({required int? userId, required String email}) async {
    if (userId == null) {
      _avatarUrl = null;
      _displayName = '';
      _avatarVersion = 0;
      notifyListeners();
      return;
    }
    final fallbackName = email.contains('@') ? email.split('@').first : email;
    try {
      final d = await _legato.getSocialProfileResilient(userId, email);
      applyFromProfile(d, fallbackName: fallbackName);
    } catch (_) {}
  }
}
