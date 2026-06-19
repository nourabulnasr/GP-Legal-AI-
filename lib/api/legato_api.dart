import 'dart:typed_data';

import 'package:legato_mobile/api/api_client.dart';
import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/config/app_config.dart';

/// Same shape as [LegatoApi.getSocialProfile] when `/api/profile/{id}` is unavailable.
Map<String, dynamic> _mapLegalProfileToSocialShape(
  Map<String, dynamic> leg,
  int userId,
  String email,
) {
  final raw = leg['profile'];
  final Map<String, dynamic> p = raw is Map<String, dynamic>
      ? raw
      : raw is Map
          ? Map<String, dynamic>.from(raw)
          : <String, dynamic>{};
  var displayName = (p['displayName'] ?? p['display_name'] ?? '').toString();
  if (displayName.isEmpty) displayName = email.split('@').first;

  List<dynamic> skills = [];
  final sk = p['skills'];
  if (sk is List) {
    skills = sk;
  } else if (sk is String) {
    skills = sk.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();
  }

  List<dynamic> experience = [];
  final ex = p['experience'];
  if (ex is List) experience = ex;

  List<dynamic> education = [];
  final ed = p['education'];
  if (ed is List) education = ed;

  return {
    'user_id': userId,
    'email': email,
    'display_name': displayName,
    'title': p['title']?.toString() ?? '',
    'company': p['company']?.toString() ?? '',
    'location': p['location']?.toString() ?? '',
    'bio': p['bio']?.toString() ?? '',
    'avatar_url': p['avatarUrl']?.toString() ?? p['avatar_url']?.toString() ?? '',
    'cover_url': p['coverUrl']?.toString() ?? p['cover_url']?.toString() ?? '',
    'skills': skills,
    'experience': experience,
    'education': education,
    'stats': {'connections': 0, 'endorsements': 0},
    'is_self': true,
  };
}

/// Mirrors `legalai-frontend/src/lib/api.ts` — same paths and form fields.
class LegatoApi {
  LegatoApi(this._api);

  final ApiClient _api;

  Future<Map<String, dynamic>> analyzeContract(
    Uint8List fileBytes,
    String filename, {
    bool useRag = true,
    bool useMl = true,
    bool useLlm = true,
    int? llmTopK,
    int? llmMaxNewTokens,
    bool save = true,
    String? query,
    bool translateToAr = false,
    bool translationOnly = false,
    String translationTargetLang = 'ar',
    bool translatePerChunkMt = false,
    String sourceLanguageMode = 'auto',
    String? sourceLanguageOverride,
  }) {
    return _api.postMultipartOcrCheck(
      fileBytes: fileBytes,
      filename: filename,
      useRag: useRag,
      useMl: useMl,
      useLlm: useLlm,
      llmTopK: llmTopK,
      llmMaxNewTokens: llmMaxNewTokens,
      save: save,
      query: query,
      translateToAr: translateToAr,
      translationOnly: translationOnly,
      translationTargetLang: translationTargetLang,
      translatePerChunkMt: translatePerChunkMt,
      sourceLanguageMode: sourceLanguageMode,
      sourceLanguageOverride: sourceLanguageOverride,
    );
  }

  /// Translation-only OCR + MT (server picks Google → LFM automatically).
  Future<Map<String, dynamic>> translateContract(
    Uint8List fileBytes,
    String filename, {
    String translationTargetLang = 'ar',
    bool save = false,
    bool translatePerChunkMt = false,
    String sourceLanguageMode = 'auto',
    String? sourceLanguageOverride,
  }) {
    return analyzeContract(
      fileBytes,
      filename,
      useRag: false,
      useMl: false,
      useLlm: false,
      save: save,
      translateToAr: true,
      translationOnly: true,
      translationTargetLang: translationTargetLang,
      translatePerChunkMt: translatePerChunkMt,
      sourceLanguageMode: sourceLanguageMode,
      sourceLanguageOverride: sourceLanguageOverride,
    );
  }

  Future<List<dynamic>> listAnalyses() => _api.getJsonList('/analyses');

  Future<Map<String, dynamic>> getAnalysis(int id) => _api.getJson('/analyses/$id');

  Future<Map<String, dynamic>> deleteAnalysis(int id) => _api.deleteJson('/analyses/$id');

  Future<List<dynamic>> adminListAll() => _api.getJsonList('/analyses/admin/all');

  Future<List<dynamic>> adminListUsers() => _api.getJsonList('/analyses/admin/users');

  Future<Map<String, dynamic>> adminUpdateUserRole(int userId, String role) {
    return _api.patchJson('/analyses/admin/users/$userId', {'role': role});
  }

  /// Updates the user's account type (e.g. 'lawyer' or 'user') in the database.
  Future<Map<String, dynamic>> adminUpdateUserType(int userId, String userType) {
    return _api.patchJson('/analyses/admin/users/$userId', {'user_type': userType});
  }

  Future<Map<String, dynamic>> adminDeleteUser(int userId) {
    return _api.deleteJson('/analyses/admin/users/$userId');
  }

  Future<List<dynamic>> adminListUserAnalyses(int userId) {
    return _api.getJsonList('/analyses/admin/user/$userId');
  }

  Future<Map<String, dynamic>> chatMessage({
    required int analysisId,
    required String message,
    List<Map<String, dynamic>>? history,
  }) {
    return _api.postJson('/chat/message', {
      'analysis_id': analysisId,
      'message': message,
      if (history != null) 'history': history,
    });
  }

  Future<Map<String, dynamic>> chatAssistant({
    required String message,
    List<Map<String, dynamic>>? history,
  }) {
    return _api.postJson('/chat/assistant', {
      'message': message,
      if (history != null) 'history': history,
    });
  }

  Future<Map<String, dynamic>> chatWithDocument({
    String? documentContext,
    int? analysisId,
    required String message,
    List<Map<String, dynamic>>? history,
  }) {
    return _api.postJsonLong(
      '/chat/document',
      {
        if (documentContext != null) 'document_context': documentContext,
        if (analysisId != null) 'analysis_id': analysisId,
        'message': message,
        if (history != null) 'history': history,
      },
      timeout: AppConfig.longTimeout,
    );
  }

  Future<Map<String, dynamic>> saveAnalysisToDb({
    required String filename,
    required String resultJson,
    String? mimeType,
    String? sha256,
    int? pageCount,
    int? ocrUsed,
    String? detectedLang,
  }) {
    return _api.postJson('/analyses', {
      'filename': filename,
      'result_json': resultJson,
      if (mimeType != null) 'mime_type': mimeType,
      if (sha256 != null) 'sha256': sha256,
      if (pageCount != null) 'page_count': pageCount,
      if (ocrUsed != null) 'ocr_used': ocrUsed,
      if (detectedLang != null) 'detected_lang': detectedLang,
    });
  }

  // --- Phase 5 /legato/* (see app/routers/legato_mobile.py) ---

  Future<Map<String, dynamic>> explainClause({
    required String clauseText,
    int? analysisId,
    String? ruleId,
    /// `ar` or `en`; omit for server-side detection from clause text.
    String? language,
  }) {
    // Local LFM can be slow on CPU; use the same timeout budget as chat.
    return _api.postJson(
      '/legato/explain-clause',
      {
      'clause_text': clauseText,
      if (analysisId != null) 'analysis_id': analysisId,
      if (ruleId != null) 'rule_id': ruleId,
      if (language != null) 'language': language,
      },
      timeout: AppConfig.chatTimeout,
    );
  }

  Future<Map<String, dynamic>> summarizeClauses({
    required List<String> clauses,
    int? analysisId,
    String? language,
  }) {
    // Local LFM can be slow on CPU; allow chat timeout.
    return _api.postJson(
      '/legato/summarize-clauses',
      {
        'clauses': clauses,
        if (analysisId != null) 'analysis_id': analysisId,
        if (language != null) 'language': language,
      },
      timeout: AppConfig.chatTimeout,
    );
  }

  Future<Map<String, dynamic>> compareContracts({
    String? textA,
    String? textB,
    int? analysisIdA,
    int? analysisIdB,
    String? language,
  }) {
    return _api.postJson('/legato/compare', {
      if (textA != null) 'text_a': textA,
      if (textB != null) 'text_b': textB,
      if (analysisIdA != null) 'analysis_id_a': analysisIdA,
      if (analysisIdB != null) 'analysis_id_b': analysisIdB,
      if (language != null) 'language': language,
    }, timeout: AppConfig.chatTimeout);
  }

  Future<Map<String, dynamic>> negotiationChat({
    required String message,
    int? analysisId,
    List<Map<String, dynamic>>? history,
  }) {
    return _api.postJsonLong('/legato/negotiation-chat', {
      'message': message,
      if (analysisId != null) 'analysis_id': analysisId,
      if (history != null) 'history': history,
    });
  }

  Future<Map<String, dynamic>> riskSummary(int analysisId) =>
      _api.getJson('/legato/risk/$analysisId');

  Future<Map<String, dynamic>> createShare(int analysisId, {int expiresDays = 30}) {
    return _api.postJson('/legato/shares', {
      'analysis_id': analysisId,
      'expires_days': expiresDays,
    });
  }

  /// Public read-only payload (no Bearer required; uses same client without token in practice).
  Future<Map<String, dynamic>> publicShare(String token) =>
      _api.getJson('/legato/shares/public/$token');

  Future<Map<String, dynamic>> createDealThread(int analysisId, {String? title}) {
    return _api.postJson('/legato/deal-threads', {
      'analysis_id': analysisId,
      if (title != null) 'title': title,
    });
  }

  Future<Map<String, dynamic>> createDealThreadByCategory({
    required String contractCategory,
    required String title,
    int? analysisId,
  }) {
    return _api.postJson('/legato/deal-threads', {
      'contract_category': contractCategory,
      'title': title,
      if (analysisId != null) 'analysis_id': analysisId,
    });
  }

  Future<Map<String, dynamic>> updateDealThreadTitle(int threadId, String title) =>
      _api.patchJson('/legato/deal-threads/$threadId', {'title': title});

  Future<Map<String, dynamic>> joinDealThread(int threadId) =>
      _api.postJson('/legato/deal-threads/$threadId/join', {});

  Future<Map<String, dynamic>> listDealThreadMembers(int threadId) =>
      _api.getJson('/legato/deal-threads/$threadId/members');

  Future<List<dynamic>> listDealThreads(int analysisId) {
    return _api.getJsonListQuery('/legato/deal-threads', {'analysis_id': '$analysisId'});
  }

  Future<List<dynamic>> listDealThreadsByCategory(String contractCategory) {
    return _api.getJsonListQuery('/legato/deal-threads', {'contract_category': contractCategory});
  }

  Future<Map<String, dynamic>> myDealCategories() => _api.getJson('/legato/my-deal-categories');

  Future<Map<String, dynamic>> dealPeers(String contractCategory) =>
      _api.getJsonQuery('/legato/deal-peers', {'contract_category': contractCategory});

  Future<Map<String, dynamic>> listConversations() => _api.getJson('/api/messages/conversations');

  Future<Map<String, dynamic>> deleteConversation(int conversationId) =>
      _api.deleteJson('/api/messages/conversations/$conversationId');

  Future<Map<String, dynamic>> getConversation(int conversationId) =>
      _api.getJson('/api/messages/conversations/$conversationId');

  Future<Map<String, dynamic>> addGroupMembers(int conversationId, List<int> memberIds) =>
      _api.postJson('/api/messages/conversations/$conversationId/members', {'member_ids': memberIds});

  Future<Map<String, dynamic>> createDirectConversation(int peerUserId) =>
      _api.postJson('/api/messages/conversations/direct', {'peer_user_id': peerUserId});

  Future<Map<String, dynamic>> createGroupConversation({
    required String title,
    required List<int> memberIds,
  }) {
    return _api.postJson('/api/messages/conversations/group', {
      'title': title,
      'member_ids': memberIds,
    });
  }

  Future<Map<String, dynamic>> listConversationMessages(int conversationId) =>
      _api.getJson('/api/messages/conversations/$conversationId/messages');

  Future<Map<String, dynamic>> postConversationMessage(int conversationId, String body) {
    return _api.postJson('/api/messages/conversations/$conversationId/messages', {'body': body});
  }

  Future<Map<String, dynamic>> postLawyerOffer(
    int conversationId, {
    required String serviceTitle,
    required String description,
    required double price,
    String currency = 'USD',
  }) {
    return _api.postJson('/api/messages/conversations/$conversationId/offer', {
      'service_title': serviceTitle,
      'description': description,
      'price': price,
      'currency': currency,
    });
  }

  Future<Map<String, dynamic>> acceptOffer(int messageId) =>
      _api.patchJson('/api/messages/$messageId/offer-status', {'status': 'accepted'});

  Future<Map<String, dynamic>> updateGroupTitle(int conversationId, String title) =>
      _api.patchJson('/api/messages/conversations/$conversationId', {'title': title});

  Future<List<dynamic>> listDealMessages(int threadId) =>
      _api.getJsonList('/legato/deal-threads/$threadId/messages');

  Future<Map<String, dynamic>> postDealMessage(int threadId, String body) {
    return _api.postJson('/legato/deal-threads/$threadId/messages', {'body': body});
  }

  Future<Map<String, dynamic>> createTimelineEvent({
    required int analysisId,
    required String label,
    required String eventDateIso,
    String source = 'manual',
  }) {
    return _api.postJson('/legato/timeline/events', {
      'analysis_id': analysisId,
      'label': label,
      'event_date': eventDateIso,
      'source': source,
    });
  }

  Future<List<dynamic>> adminTimelineAll() => _api.getJsonList('/legato/timeline/all');

  Future<List<dynamic>> timelineMe() => _api.getJsonList('/legato/timeline/me');

  Future<void> deleteTimelineEvent(int eventId) async {
    await _api.deleteJson('/legato/timeline/events/$eventId');
  }

  Future<Map<String, dynamic>> getLegalProfile() => _api.getJson('/legato/profile/me');

  Future<Map<String, dynamic>> putLegalProfile(Map<String, dynamic> fields) =>
      _api.putJson('/legato/profile/me', fields);

  Future<List<dynamic>> listNetworkProfiles() => _api.getJsonList('/legato/network/profiles');

  Future<Map<String, dynamic>> recordSignature({
    required int analysisId,
    required String signerName,
    required bool consentAcknowledged,
    String? signaturePngBase64,
  }) {
    return _api.postJson('/legato/signatures', {
      'analysis_id': analysisId,
      'signer_name': signerName,
      'consent_acknowledged': consentAcknowledged,
      if (signaturePngBase64 != null) 'signature_png_base64': signaturePngBase64,
    });
  }

  // --- Social / professional networking (`/api/*`) ---

  Future<Map<String, dynamic>> getPost(int postId) => _api.getJson('/api/posts/$postId');

  Future<Map<String, dynamic>> getPosts({String? category, int page = 1, int pageSize = 20}) {
    return _api.getJsonQuery('/api/posts', {
      'page': '$page',
      'page_size': '$pageSize',
      if (category != null && category.isNotEmpty && category != 'All Updates') 'category': category,
    });
  }

  Future<Map<String, dynamic>> createPost({
    required String content,
    List<String>? tags,
    String category = 'All Updates',
    Uint8List? imageBytes,
    String? imageFilename,
  }) {
    return _api.postMultipartPost(
      '/api/posts',
      fields: {
        'content': content,
        'category': category,
        'tags': (tags ?? []).join(','),
      },
      imageBytes: imageBytes,
      imageFilename: imageFilename,
    );
  }

  Future<Map<String, dynamic>> togglePostLike(int postId) => _api.postJson('/api/posts/$postId/like', {});

  Future<Map<String, dynamic>> addPostComment(int postId, String content) =>
      _api.postJson('/api/posts/$postId/comment', {'content': content});

  Future<Map<String, dynamic>> getPostComments(int postId, {int page = 1}) {
    return _api.getJsonQuery('/api/posts/$postId/comments', {'page': '$page', 'page_size': '30'});
  }

  Future<Map<String, dynamic>> sharePost(int postId) => _api.postJson('/api/posts/$postId/share', {});

  Future<Map<String, dynamic>> getSocialProfile(int userId) => _api.getJson('/api/profile/$userId');

  /// Tries `/api/profile/{id}`; on 404 or missing route, merges `/legato/profile/me` into the same shape.
  Future<Map<String, dynamic>> getSocialProfileResilient(int userId, String email) async {
    try {
      return await getSocialProfile(userId);
    } on ApiException catch (e) {
      final msg = e.message.toLowerCase();
      if (e.statusCode == 404 || msg.contains('not found') || msg.contains('user not found')) {
        final leg = await getLegalProfile();
        return _mapLegalProfileToSocialShape(leg, userId, email);
      }
      rethrow;
    }
  }

  /// Merges into `LegatoProfile` JSON via `/api/profile/me` (displayName, title, skills, bio, …).
  Future<Map<String, dynamic>> putApiProfile(Map<String, dynamic> fields) =>
      _api.putJson('/api/profile/me', fields);

  /// Same data as [putApiProfile]; if `/api/profile/me` is missing (404), writes merged JSON via `/legato/profile/me`.
  Future<Map<String, dynamic>> putProfileResilient(Map<String, dynamic> fields) async {
    try {
      return await putApiProfile(fields);
    } on ApiException catch (e) {
      final msg = e.message.toLowerCase();
      if (e.statusCode == 404 || msg.contains('not found')) {
        final leg = await getLegalProfile();
        final cur = Map<String, dynamic>.from((leg['profile'] as Map?) ?? {});
        for (final entry in fields.entries) {
          cur[entry.key] = entry.value;
        }
        return await putLegalProfile(cur);
      }
      rethrow;
    }
  }

  Future<Map<String, dynamic>> addProfileEducation({
    required String school,
    String degree = '',
    String year = '',
  }) {
    return _api.postJson('/api/profile/me/education', {
      'school': school,
      'degree': degree,
      'year': year,
    });
  }

  Future<Map<String, dynamic>> addProfileExperience({
    required String title,
    String company = '',
    String startDate = '',
    String? endDate,
    String description = '',
  }) {
    return _api.postJson('/api/profile/me/experience', {
      'title': title,
      'company': company,
      'startDate': startDate,
      if (endDate != null && endDate.isNotEmpty) 'endDate': endDate,
      'description': description,
    });
  }

  Future<Map<String, dynamic>> updateProfileExperience(
    int index, {
    required String title,
    String company = '',
    String startDate = '',
    String? endDate,
    String description = '',
  }) {
    return _api.putJson('/api/profile/me/experience/$index', {
      'title': title,
      'company': company,
      'start_date': startDate,
      'end_date': endDate ?? '',
      'description': description,
    });
  }

  Future<void> deleteProfileExperience(int index) async {
    await _api.deleteJson('/api/profile/me/experience/$index');
  }

  Future<Map<String, dynamic>> updateProfileEducation(
    int index, {
    required String school,
    String degree = '',
    String year = '',
  }) {
    return _api.putJson('/api/profile/me/education/$index', {
      'school': school,
      'degree': degree,
      'year': year,
    });
  }

  Future<void> deleteProfileEducation(int index) async {
    await _api.deleteJson('/api/profile/me/education/$index');
  }

  Future<Map<String, dynamic>> uploadProfileAvatar(Uint8List bytes, String filename) {
    return _api.postMultipartBytes(
      '/api/profile/me/avatar',
      bytes: bytes,
      filename: filename,
      fieldName: 'avatar',
    );
  }

  Future<Map<String, dynamic>> deletePost(int postId) =>
      _api.deleteJson('/api/posts/$postId');

  Future<Map<String, dynamic>> getNetworkStats() => _api.getJson('/api/network/stats');

  Future<Map<String, dynamic>> getNetworkSuggestions() =>
      _api.getJsonQuery('/api/network/suggestions', {'limit': '12'});

  Future<Map<String, dynamic>> searchNetwork(String q, {int limit = 20}) =>
      _api.getJsonQuery('/api/network/search', {'q': q, 'limit': '$limit'});

  Future<Map<String, dynamic>> sendNetworkInvite(int toUserId) =>
      _api.postJson('/api/network/invites', {'to_user_id': toUserId});

  Future<Map<String, dynamic>> submitPostProposal({
    required int postId,
    required String message,
    required double hourlyRate,
  }) =>
      _api.postJson('/api/posts/$postId/proposals', {
        'message': message,
        'hourly_rate': hourlyRate,
      });

  Future<List<dynamic>> getPostProposals(int postId) async {
    final r = await _api.getJson('/api/posts/$postId/proposals');
    return (r['items'] as List<dynamic>?) ?? [];
  }

  Future<Map<String, dynamic>> respondToPostProposal({
    required int proposalId,
    required String action,
    double? counterRate,
  }) =>
      _api.patchJson('/api/posts/proposals/$proposalId', {
        'action': action,
        if (counterRate != null) 'counter_rate': counterRate,
      });

  Future<Map<String, dynamic>> getPendingInvites() => _api.getJson('/api/network/invites');

  Future<Map<String, dynamic>> getNetworkConnections() =>
      _api.getJson('/api/network/connections');

  Future<Map<String, dynamic>> acceptNetworkInvite(int inviteId) =>
      _api.postJson('/api/network/invites/$inviteId/accept', {});

  Future<Map<String, dynamic>> getEndorsements(int userId) =>
      _api.getJson('/api/profile/$userId/endorsements');

  Future<Map<String, dynamic>> endorseSkill(int userId, String skill) =>
      _api.postJson('/api/profile/$userId/endorse', {'skill': skill});

  Future<Map<String, dynamic>> getProfileDocuments(int userId) =>
      _api.getJson('/api/profile/$userId/documents');

  Future<Map<String, dynamic>> addProfileDocument({required String title, required String fileUrl}) =>
      _api.postJson('/api/profile/me/documents', {'title': title, 'file_url': fileUrl});

  Future<Map<String, dynamic>> uploadProfileDocument({
    required String title,
    required String filePath,
    required String filename,
  }) {
    return _api.postMultipart(
      '/api/profile/me/documents/upload',
      filePath: filePath,
      fieldName: 'file',
      filename: filename,
      fields: {'title': title},
    );
  }

  Future<Map<String, dynamic>> getRecommendations(int userId) =>
      _api.getJson('/api/profile/$userId/recommendations');

  Future<Map<String, dynamic>> addRecommendation(int userId, String content) =>
      _api.postJson('/api/profile/$userId/recommendations', {'content': content});

  // --- Activity notifications (feed → alerts) ---

  Future<Map<String, dynamic>> listNotifications({int page = 1, int pageSize = 40}) =>
      _api.getJsonQuery('/api/notifications', {'page': '$page', 'page_size': '$pageSize'});

  Future<int> unreadNotificationCount() async {
    final r = await _api.getJson('/api/notifications/unread-count');
    return (r['unread_count'] as num?)?.toInt() ?? 0;
  }

  Future<void> markNotificationRead(int id) async {
    await _api.postJson('/api/notifications/$id/read', {});
  }

  Future<void> markAllNotificationsRead() async {
    await _api.postJson('/api/notifications/read-all', {});
  }

  Future<void> deleteNotification(int id) async {
    await _api.deleteJson('/api/notifications/$id');
  }

  // --- Lawyer verification ---

  Future<Map<String, dynamic>> lawyerApply({
    String barLicenseNumber = '',
    int? yearsOfExperience,
    double? hourlyRate,
    Uint8List? documentBytes,
    String? documentFilename,
    Uint8List? cvBytes,
    String? cvFilename,
    Uint8List? idCardBytes,
    String? idCardFilename,
    Uint8List? idCardBackBytes,
    String? idCardBackFilename,
  }) {
    return _api.postMultipartLawyerApply(
      barLicenseNumber: barLicenseNumber,
      yearsOfExperience: yearsOfExperience,
      hourlyRate: hourlyRate,
      documentBytes: documentBytes,
      documentFilename: documentFilename,
      cvBytes: cvBytes,
      cvFilename: cvFilename,
      idCardBytes: idCardBytes,
      idCardFilename: idCardFilename,
      idCardBackBytes: idCardBackBytes,
      idCardBackFilename: idCardBackFilename,
    );
  }

  Future<Map<String, dynamic>> adminUpdateLawyerStatus(int userId, String lawyerStatus) {
    return _api.patchJson('/analyses/admin/users/$userId', {'lawyer_status': lawyerStatus});
  }

  Future<Map<String, dynamic>> lawyerStatus() => _api.getJson('/lawyer/status');

  Future<Uint8List> myLawyerCv() => _api.getBytes('/lawyer/cv');

  Future<Uint8List> myLawyerIdCard() => _api.getBytes('/lawyer/id-card');

  Future<Uint8List> adminDownloadLawyerCv(int applicationId) =>
      _api.getBytes('/admin/lawyers/cv/$applicationId');

  Future<Uint8List> adminDownloadLawyerIdCard(int applicationId) =>
      _api.getBytes('/admin/lawyers/id-card/$applicationId');

  Future<Uint8List> adminDownloadLawyerIdCardBack(int applicationId) =>
      _api.getBytes('/admin/lawyers/id-card-back/$applicationId');

  Future<List<dynamic>> adminListLawyerApplications({String status = 'pending'}) {
    return _api.getJsonList('/admin/lawyers?status=$status');
  }

  Future<Map<String, dynamic>> adminReviewLawyerApplication(
    int applicationId, {
    required String action,
    String? adminNote,
    double? negotiatedHourlyRate,
  }) {
    return _api.patchJson('/admin/lawyers/$applicationId/review', {
      'action': action,
      if (adminNote != null && adminNote.isNotEmpty) 'admin_note': adminNote,
      if (negotiatedHourlyRate != null) 'negotiated_hourly_rate': negotiatedHourlyRate,
    });
  }

  Future<List<dynamic>> getVerifiedLawyers() async {
    final r = await _api.getJson('/api/network/lawyers');
    return (r['items'] as List<dynamic>?) ?? [];
  }

  Future<Map<String, dynamic>> requestConsultation({
    required int lawyerId,
    required int durationMinutes,
    required String scheduledAt,
    String? notes,
  }) {
    return _api.postJson('/api/consultations', {
      'lawyer_id': lawyerId,
      'duration_minutes': durationMinutes,
      'scheduled_at': scheduledAt,
      if (notes != null && notes.isNotEmpty) 'notes': notes,
    });
  }

  Future<Map<String, dynamic>> checkoutConsultationPayment(int consultationId) =>
      _api.postJson('/api/consultations/$consultationId/payment/checkout', {});

  Future<Map<String, dynamic>> getConsultation(int id) =>
      _api.getJson('/api/consultations/$id');

  Future<Map<String, dynamic>> respondConsultation(int id, {required bool accept}) {
    return _api.postJson('/api/consultations/$id/respond', {
      'action': accept ? 'accept' : 'reject',
    });
  }

  Future<Map<String, dynamic>> getConsultationSession(int conversationId) =>
      _api.getJson('/api/consultations/session/conversation/$conversationId');

  Future<Map<String, dynamic>> respondLawyerRate({required bool accept}) =>
      _api.postJson('/lawyer/rate-response', {'accept': accept});
}
