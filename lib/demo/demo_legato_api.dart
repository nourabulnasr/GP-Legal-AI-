import 'dart:typed_data';

import 'package:legato_mobile/api/api_client.dart';
import 'package:legato_mobile/api/legato_api.dart';
import 'package:legato_mobile/demo/demo_state.dart';

/// Drop-in replacement for [LegatoApi] that reads/writes [DemoState].
/// Write operations (like, comment, review) mutate in-memory state so the
/// demo stays consistent across persona switches within the same app session.
class DemoLegatoApi extends LegatoApi {
  DemoLegatoApi() : super(ApiClient());

  // ── Lawyer verification ────────────────────────────────────────────────────

  @override
  Future<Map<String, dynamic>> lawyerStatus() async => DemoState.lawyerStatus;

  /// Also stores uploaded files so admin can view/download them.
  @override
  Future<Map<String, dynamic>> lawyerApply({
    String barLicenseNumber = '',
    int? yearsOfExperience,
    Uint8List? documentBytes,
    String? documentFilename,
    Uint8List? cvBytes,
    String? cvFilename,
    Uint8List? idCardBytes,
    String? idCardFilename,
  }) async {
    DemoState.submitLawyerApp(
      userId: DemoState.current.id,
      cvFilename: cvFilename ?? documentFilename,
      cvBytes: cvBytes ?? documentBytes,
      idCardFilename: idCardFilename,
      idCardBytes: idCardBytes,
      barLicenseNumber: barLicenseNumber.isEmpty ? null : barLicenseNumber,
      yearsOfExperience: yearsOfExperience,
    );
    return {'status': 'pending', 'message': 'Application submitted (demo)'};
  }

  @override
  Future<List<dynamic>> adminListLawyerApplications({String status = 'pending'}) async =>
      DemoState.lawyerApps(status: status);

  /// Updates [DemoState] so lawyer status is reflected on next login.
  @override
  Future<Map<String, dynamic>> adminReviewLawyerApplication(
    int applicationId, {
    required String action,
    String? adminNote,
  }) async {
    DemoState.reviewLawyerApp(applicationId, action, adminNote);
    return {
      'status': action == 'approve' ? 'approved' : 'rejected',
      'message': 'Reviewed (demo)',
    };
  }

  /// Returns the real bytes the lawyer uploaded, or empty if none were uploaded.
  @override
  Future<Uint8List> adminDownloadLawyerCv(int applicationId) async =>
      DemoState.cvBytesForApp(applicationId) ?? Uint8List(0);

  @override
  Future<Uint8List> adminDownloadLawyerIdCard(int applicationId) async =>
      DemoState.idCardBytesForApp(applicationId) ?? Uint8List(0);

  // ── Notifications (dynamic — changes after admin review) ──────────────────

  @override
  Future<Map<String, dynamic>> listNotifications({int page = 1, int pageSize = 40}) async =>
      {'items': DemoState.notifications};

  @override
  Future<int> unreadNotificationCount() async =>
      DemoState.notifications.where((n) => n['is_read'] == false).length;

  @override
  Future<void> markNotificationRead(int id) async {}

  @override
  Future<void> markAllNotificationsRead() async {}

  @override
  Future<void> deleteNotification(int id) async {}

  // ── Feed / posts ──────────────────────────────────────────────────────────

  @override
  Future<Map<String, dynamic>> getPosts({String? category, int page = 1, int pageSize = 20}) async {
    final all = DemoState.posts;
    final items =
        category == null ? all : all.where((p) => p['category'] == category).toList();
    return {'items': items, 'total': items.length};
  }

  @override
  Future<Map<String, dynamic>> getPost(int postId) async =>
      DemoState.posts.firstWhere((p) => p['id'] == postId,
          orElse: () => DemoState.posts.first);

  @override
  Future<Map<String, dynamic>> togglePostLike(int postId) async => {'liked': true};

  @override
  Future<Map<String, dynamic>> addPostComment(int postId, String content) async => {
        'id': 99,
        'content': content,
        'author_display_name': DemoState.nameOf(DemoState.current),
      };

  @override
  Future<Map<String, dynamic>> getPostComments(int postId, {int page = 1}) async =>
      {'items': <dynamic>[], 'total': 0};

  @override
  Future<Map<String, dynamic>> sharePost(int postId) async =>
      {'message': 'Shared (demo)'};

  @override
  Future<Map<String, dynamic>> createPost({
    required String content,
    List<String>? tags,
    String category = 'All Updates',
    Uint8List? imageBytes,
    String? imageFilename,
  }) async =>
      {'id': 100, 'content': content, 'category': category};

  @override
  Future<Map<String, dynamic>> deletePost(int postId) async =>
      {'message': 'Deleted (demo)'};

  // ── Network / profiles ────────────────────────────────────────────────────

  @override
  Future<Map<String, dynamic>> getNetworkStats() async => {
        'connections_count': 12,
        'connections': 12,
        'profile_views': 45,
        'search_appearances': 8,
      };

  @override
  Future<Map<String, dynamic>> getNetworkSuggestions() async =>
      {'items': DemoState.networkSuggestions};

  @override
  Future<Map<String, dynamic>> getNetworkConnections() async =>
      {'items': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> getPendingInvites() async => {'items': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> searchNetwork(String q, {int limit = 20}) async =>
      {'items': DemoState.networkSuggestions.where((s) {
        final name = (s['display_name'] ?? '').toString().toLowerCase();
        return name.contains(q.toLowerCase());
      }).toList()};

  @override
  Future<Map<String, dynamic>> sendNetworkInvite(int toUserId) async =>
      {'message': 'Invite sent (demo)'};

  @override
  Future<Map<String, dynamic>> acceptNetworkInvite(int inviteId) async =>
      {'message': 'Accepted (demo)'};

  @override
  Future<Map<String, dynamic>> getLegalProfile() async {
    final u = DemoState.current;
    return {
      'profile': {
        'displayName': DemoState.nameOf(u),
        'display_name': DemoState.nameOf(u),
        'title': DemoState.titleOf(u),
        'company': u.isAdmin ? '' : 'Al-Rashid Legal Group',
        'location': 'Cairo, Egypt',
        'bio': u.isAdmin
            ? 'Platform administrator.'
            : 'Legal professional specialising in contract law.',
        'avatarUrl': null,
        'avatar_url': null,
        'skills': u.isLawyerAccount
            ? ['Contract Law', 'Corporate Law', 'Dispute Resolution']
            : ['Legal Research'],
        'experience': <dynamic>[],
        'education': <dynamic>[],
      }
    };
  }

  @override
  Future<Map<String, dynamic>> putLegalProfile(Map<String, dynamic> fields) async =>
      {'message': 'Saved (demo)'};

  @override
  Future<Map<String, dynamic>> putApiProfile(Map<String, dynamic> fields) async =>
      {'message': 'Saved (demo)'};

  @override
  Future<Map<String, dynamic>> putProfileResilient(Map<String, dynamic> fields) async =>
      {'message': 'Saved (demo)'};

  @override
  Future<Map<String, dynamic>> getSocialProfile(int userId) async {
    final u = DemoState.current;
    if (userId == u.id) return DemoState.profileOf(u);
    final match = DemoState.networkSuggestions.firstWhere(
      (s) => (s['user_id'] as int?) == userId,
      orElse: () => <String, dynamic>{},
    );
    if (match.isNotEmpty) return Map<String, dynamic>.from(match);
    return {
      'user_id': userId,
      'display_name': 'Legal Professional',
      'title': 'Attorney',
      'company': 'Legal Firm',
      'location': 'Egypt',
      'bio': '',
      'avatar_url': null,
      'user_type': 'user',
      'lawyer_status': null,
      'is_verified_lawyer': false,
    };
  }

  @override
  Future<Map<String, dynamic>> getSocialProfileResilient(int userId, String email) =>
      getSocialProfile(userId);

  @override
  Future<Map<String, dynamic>> uploadProfileAvatar(Uint8List bytes, String filename) async =>
      {'avatar_url': null};

  @override
  Future<Map<String, dynamic>> getEndorsements(int userId) async =>
      {'endorsements': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> endorseSkill(int userId, String skill) async =>
      {'message': 'Endorsed (demo)'};

  @override
  Future<Map<String, dynamic>> getProfileDocuments(int userId) async =>
      {'documents': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> getRecommendations(int userId) async =>
      {'recommendations': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> addRecommendation(int userId, String content) async =>
      {'message': 'Added (demo)'};

  @override
  Future<Map<String, dynamic>> addProfileEducation(
          {required String school, String degree = '', String year = ''}) async =>
      {'message': 'Added (demo)'};

  @override
  Future<Map<String, dynamic>> addProfileExperience({
    required String title,
    String company = '',
    String startDate = '',
    String? endDate,
    String description = '',
  }) async =>
      {'message': 'Added (demo)'};

  @override
  Future<Map<String, dynamic>> updateProfileExperience(
    int index, {
    required String title,
    String company = '',
    String startDate = '',
    String? endDate,
    String description = '',
  }) async =>
      {'message': 'Updated (demo)'};

  @override
  Future<void> deleteProfileExperience(int index) async {}

  @override
  Future<Map<String, dynamic>> updateProfileEducation(
    int index, {
    required String school,
    String degree = '',
    String year = '',
  }) async =>
      {'message': 'Updated (demo)'};

  @override
  Future<void> deleteProfileEducation(int index) async {}

  // ── Admin users ───────────────────────────────────────────────────────────

  @override
  Future<List<dynamic>> adminListAll() async => <dynamic>[];

  @override
  Future<List<dynamic>> adminListUsers() async => DemoState.adminUsers;

  @override
  Future<List<dynamic>> adminListUserAnalyses(int userId) async => <dynamic>[];

  @override
  Future<Map<String, dynamic>> adminUpdateUserRole(int userId, String role) async =>
      {'message': 'Role updated (demo)'};

  @override
  Future<Map<String, dynamic>> adminUpdateUserType(int userId, String userType) async =>
      {'message': 'User type updated (demo)'};

  @override
  Future<Map<String, dynamic>> adminUpdateLawyerStatus(int userId, String lawyerStatus) async {
    DemoState.reviewLawyerApp(userId, lawyerStatus == 'approved' ? 'approve' : 'reject', null);
    return {'message': 'Lawyer status updated (demo)'};
  }

  @override
  Future<Map<String, dynamic>> adminDeleteUser(int userId) async =>
      {'message': 'User deleted (demo)'};

  // ── Analyses ──────────────────────────────────────────────────────────────

  @override
  Future<List<dynamic>> listAnalyses() async => <dynamic>[];

  // ── Timeline ──────────────────────────────────────────────────────────────

  @override
  Future<List<dynamic>> timelineMe() async => <dynamic>[];

  @override
  Future<List<dynamic>> adminTimelineAll() async => <dynamic>[];

  // ── Messages ──────────────────────────────────────────────────────────────

  @override
  Future<Map<String, dynamic>> listConversations() async =>
      {'conversations': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> createDirectConversation(int peerUserId) async =>
      {'id': 99, 'title': null, 'messages': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> createGroupConversation(
          {required String title, required List<int> memberIds}) async =>
      {'id': 100, 'title': title};

  @override
  Future<Map<String, dynamic>> getConversation(int conversationId) async =>
      {'id': conversationId, 'messages': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> listConversationMessages(int conversationId) async =>
      {'messages': <dynamic>[]};

  @override
  Future<Map<String, dynamic>> postConversationMessage(
          int conversationId, String body) async =>
      {'id': 1, 'body': body};

  // ── Legato extras ─────────────────────────────────────────────────────────

  @override
  Future<Map<String, dynamic>> myDealCategories() async =>
      {'categories': <dynamic>[]};

  @override
  Future<List<dynamic>> listNetworkProfiles() async => DemoState.networkSuggestions;

  @override
  Future<Map<String, dynamic>> recordSignature({
    required int analysisId,
    required String signerName,
    required bool consentAcknowledged,
    String? signaturePngBase64,
  }) async =>
      {'message': 'Signature recorded (demo)'};
}
