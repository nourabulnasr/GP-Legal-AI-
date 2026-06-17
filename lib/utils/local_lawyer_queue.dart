import 'dart:convert';
import 'dart:typed_data';
import 'package:shared_preferences/shared_preferences.dart';

/// Local (SharedPreferences / localStorage) queue of lawyer applications.
///
/// Used as a fallback when the backend doesn't have the dedicated
/// `/lawyer/apply` and `/admin/lawyers` endpoints.  All data lives
/// in the browser's localStorage, so it survives logout → login cycles
/// on the same device — exactly the scenario needed for a demo.
class LocalLawyerQueue {
  static const _key = 'legato_pending_lawyer_apps';

  static Future<List<Map<String, dynamic>>> getAll() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_key);
    if (raw == null) return [];
    try {
      return (jsonDecode(raw) as List).cast<Map<String, dynamic>>();
    } catch (_) {
      return [];
    }
  }

  static Future<List<Map<String, dynamic>>> getPending() async {
    final all = await getAll();
    return all.where((m) => m['status'] == 'pending').toList();
  }

  static Future<void> addOrUpdate({
    required int userId,
    required String email,
    String? cvFilename,
    Uint8List? cvBytes,
    String? idCardFilename,
    Uint8List? idCardBytes,
    int? yearsOfExperience,
    String? barLicenseNumber,
  }) async {
    final all = await getAll();
    // Replace any existing entry for this user so re-applications update it.
    all.removeWhere((m) => (m['user_id'] as int?) == userId);
    all.add({
      'id': userId,
      'user_id': userId,
      'user_email': email,
      'status': 'pending',
      'cv_filename': cvFilename ?? '',
      'cv_base64': cvBytes != null ? base64Encode(cvBytes) : null,
      'id_card_filename': idCardFilename ?? '',
      'id_card_base64': idCardBytes != null ? base64Encode(idCardBytes) : null,
      'years_of_experience': yearsOfExperience,
      'bar_license_number': barLicenseNumber ?? '',
      'has_cv': (cvFilename != null && cvFilename.isNotEmpty),
      'has_id_card': (idCardFilename != null && idCardFilename.isNotEmpty),
      'has_document': false,
      'document_filename': '',
      'admin_note': '',
      'submitted_at': DateTime.now().toIso8601String(),
    });
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(all));
  }

  static Future<Uint8List?> getFileBytes(int userId, String fileType) async {
    final all = await getAll();
    Map<String, dynamic>? entry;
    for (final m in all) {
      if ((m['user_id'] as int?) == userId) { entry = m; break; }
    }
    if (entry == null) return null;
    final b64 = fileType == 'cv' ? entry['cv_base64'] : entry['id_card_base64'];
    if (b64 == null || (b64 as String).isEmpty) return null;
    try { return base64Decode(b64); } catch (_) { return null; }
  }

  static Future<void> setStatus(int userId, String status, {String? adminNote}) async {
    final all = await getAll();
    for (final m in all) {
      if ((m['user_id'] as int?) == userId) {
        m['status'] = status;
        if (adminNote != null) m['admin_note'] = adminNote;
      }
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(all));
  }

  static Future<void> remove(int userId) async {
    final all = await getAll();
    all.removeWhere((m) => (m['user_id'] as int?) == userId);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(all));
  }
}
