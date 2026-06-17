import 'dart:typed_data';

import 'package:legato_mobile/models/user_model.dart';

// ── Private data class ────────────────────────────────────────────────────────

class _LawyerApp {
  _LawyerApp({
    required this.userId,
    required this.status,
    this.cvFilename,
    this.cvBytes,
    this.idCardFilename,
    this.idCardBytes,
    this.barLicenseNumber,
    this.yearsOfExperience,
    String? submittedAt,
  }) : submittedAt = submittedAt ?? DateTime.now().toIso8601String();

  final int userId;
  String status;          // 'pending' | 'approved' | 'rejected'
  String? cvFilename;
  Uint8List? cvBytes;     // real bytes when user actually uploaded a file
  String? idCardFilename;
  Uint8List? idCardBytes;
  String? barLicenseNumber;
  int? yearsOfExperience;
  String? adminNote;
  final String submittedAt;
}

// ── Shared demo state ─────────────────────────────────────────────────────────

/// Single source of truth for the entire demo layer.
/// Static fields survive persona switches (logout → login) for the app lifetime.
class DemoState {
  DemoState._();

  // ── Base user templates ───────────────────────────────────────────────────

  static const _adminBase = UserModel(id: 1, email: 'admin@demo.com', role: 'admin', userType: 'user');
  static const _pendingBase = UserModel(id: 2, email: 'lawyer@demo.com', role: 'user', userType: 'lawyer', lawyerStatus: 'pending');
  static const _approvedBase = UserModel(id: 3, email: 'verified@demo.com', role: 'user', userType: 'lawyer', lawyerStatus: 'approved');
  static const _rejectedBase = UserModel(id: 4, email: 'rejected@demo.com', role: 'user', userType: 'lawyer', lawyerStatus: 'rejected');
  static const _userBase = UserModel(id: 5, email: 'user@demo.com', role: 'user', userType: 'user');

  // ── Mutable state (persists across persona switches) ──────────────────────

  static UserModel _current = _adminBase;

  /// Lawyer status overrides set by admin review. Maps userId → new status.
  static final Map<int, String> _statusOverrides = {};

  /// Lawyer application records. Pre-seeded with a fake pending app for userId 2
  /// so the admin persona sees something on first launch.
  static final Map<int, _LawyerApp> _apps = {
    2: _LawyerApp(
      userId: 2,
      status: 'pending',
      cvFilename: 'cv_ahmed_karimi.pdf',
      cvBytes: null,
      idCardFilename: 'national_id_ahmed.jpg',
      idCardBytes: null,
      barLicenseNumber: 'BAR-2024-001',
      submittedAt: '2024-01-14T09:00:00Z',
    ),
  };

  // ── Current user ──────────────────────────────────────────────────────────

  static UserModel get current => _current;

  /// Resolve a base UserModel against any admin review override.
  static UserModel _resolved(UserModel base) {
    final override = _statusOverrides[base.id];
    if (override != null && base.isLawyerAccount) {
      return UserModel(id: base.id, email: base.email, role: base.role, userType: base.userType, lawyerStatus: override);
    }
    return base;
  }

  static void setUser(UserModel u) => _current = _resolved(u);

  static UserModel forEmail(String email) {
    final e = email.toLowerCase().trim();
    UserModel base;
    if (e.startsWith('admin')) {
      base = _adminBase;
    } else if (e.startsWith('lawyer') || e.startsWith('pending')) {
      base = _pendingBase;
    } else if (e.startsWith('verified') || e.startsWith('approved')) {
      base = _approvedBase;
    } else if (e.startsWith('rejected')) {
      base = _rejectedBase;
    } else {
      base = _userBase;
    }
    return _resolved(base);
  }

  // ── Display helpers ───────────────────────────────────────────────────────

  static String nameOf(UserModel u) => switch (u.id) {
        1 => 'Admin',
        2 => 'Ahmed Karimi',
        3 => 'Sarah Al-Hassan',
        4 => 'Omar Farouk',
        _ => 'Demo User',
      };

  static String titleOf(UserModel u) {
    if (u.isAdmin) return 'System Administrator';
    if (u.isVerifiedLawyer) return 'Senior Legal Counsel';
    if (u.isLawyerAccount) return 'Legal Counsel';
    return 'Legal Professional';
  }

  // ── Lawyer application actions ────────────────────────────────────────────

  /// Called when a lawyer submits their docs (via registration or application screen).
  /// Overwrites any previous submission for that user.
  static void submitLawyerApp({
    required int userId,
    String? cvFilename,
    Uint8List? cvBytes,
    String? idCardFilename,
    Uint8List? idCardBytes,
    String? barLicenseNumber,
    int? yearsOfExperience,
  }) {
    // Reset any admin review so the new submission is pending
    _statusOverrides.remove(userId);
    _apps[userId] = _LawyerApp(
      userId: userId,
      status: 'pending',
      cvFilename: cvFilename,
      cvBytes: cvBytes,
      idCardFilename: idCardFilename,
      idCardBytes: idCardBytes,
      barLicenseNumber: barLicenseNumber?.isEmpty == true ? null : barLicenseNumber,
      yearsOfExperience: yearsOfExperience,
    );
    // If the current user is the one who just applied, update their model
    if (_current.id == userId) {
      _current = UserModel(id: _current.id, email: _current.email, role: _current.role, userType: _current.userType, lawyerStatus: 'pending');
    }
  }

  /// Called by admin when they approve or reject an application.
  static void reviewLawyerApp(int appId, String action, String? adminNote) {
    final newStatus = action == 'approve' ? 'approved' : 'rejected';
    _statusOverrides[appId] = newStatus;
    final app = _apps[appId];
    if (app != null) {
      app.status = newStatus;
      app.adminNote = adminNote;
    }
    // Keep current user in sync if they happen to be logged in as the reviewed user
    if (_current.id == appId) {
      _current = UserModel(id: _current.id, email: _current.email, role: _current.role, userType: _current.userType, lawyerStatus: newStatus);
    }
  }

  // ── Lawyer status (for the banner) ───────────────────────────────────────

  static Map<String, dynamic> get lawyerStatus {
    final u = _current;
    final app = _apps[u.id];
    if (app != null) {
      return {
        'status': app.status,
        'cv_filename': app.cvFilename,
        'id_card_filename': app.idCardFilename,
        'bar_license_number': app.barLicenseNumber ?? 'BAR-2024-001',
        'years_of_experience': app.yearsOfExperience,
        'admin_note': app.adminNote,
      };
    }
    // No stored app → derive from the user's base lawyerStatus
    final name = nameOf(u).toLowerCase().replaceAll(' ', '_');
    return switch (u.lawyerStatus) {
      'approved' => {'status': 'approved', 'cv_filename': 'cv_$name.pdf', 'id_card_filename': 'national_id.jpg', 'bar_license_number': 'BAR-2023-047', 'admin_note': null},
      'rejected' => {'status': 'rejected', 'cv_filename': 'cv_$name.pdf', 'id_card_filename': 'national_id.jpg', 'bar_license_number': 'BAR-2024-003', 'admin_note': 'Submitted documents did not meet verification requirements. Please resubmit.'},
      _ => {'status': 'not_applied', 'cv_filename': null, 'id_card_filename': null, 'bar_license_number': null, 'admin_note': null},
    };
  }

  // ── Admin: lawyer applications list ──────────────────────────────────────

  static List<Map<String, dynamic>> lawyerApps({String status = 'pending'}) {
    return _apps.values
        .where((a) => status == 'all' || a.status == status)
        .map((a) => {
              'id': a.userId,
              'user_id': a.userId,
              'user_email': _emailFor(a.userId),
              'user_display_name': nameOf(_baseFor(a.userId)),
              'status': a.status,
              'cv_filename': a.cvFilename,
              'id_card_filename': a.idCardFilename,
              'bar_license_number': a.barLicenseNumber ?? 'BAR-2024-001',
              'years_of_experience': a.yearsOfExperience,
              'submitted_at': a.submittedAt,
              'admin_note': a.adminNote,
            })
        .toList();
  }

  static Uint8List? cvBytesForApp(int appId) => _apps[appId]?.cvBytes;
  static Uint8List? idCardBytesForApp(int appId) => _apps[appId]?.idCardBytes;

  // ── Notifications ─────────────────────────────────────────────────────────

  static List<Map<String, dynamic>> get notifications {
    final u = _current;
    if (u.isAdmin) {
      // One notification per pending application
      return _apps.values
          .where((a) => a.status == 'pending')
          .map((a) => {
                'id': a.userId,
                'type': 'lawyer_application',
                'actor_name': nameOf(_baseFor(a.userId)),
                'text': 'submitted a lawyer verification application',
                'created_at': a.submittedAt,
                'is_read': false,
                'reference_id': a.userId,
              })
          .toList();
    }
    final app = _apps[u.id];
    if (app != null && app.status == 'rejected') {
      return [
        {
          'id': 100 + u.id,
          'type': 'lawyer_review',
          'actor_name': 'Admin',
          'text': 'Your lawyer verification application has been reviewed.',
          'created_at': DateTime.now().toIso8601String(),
          'is_read': false,
          'reference_id': null,
        }
      ];
    }
    if (app != null && app.status == 'approved') {
      return [
        {
          'id': 200 + u.id,
          'type': 'lawyer_review',
          'actor_name': 'Admin',
          'text': 'Congratulations! Your lawyer verification has been approved.',
          'created_at': DateTime.now().toIso8601String(),
          'is_read': false,
          'reference_id': null,
        }
      ];
    }
    return [];
  }

  // ── Admin users list ──────────────────────────────────────────────────────

  static List<Map<String, dynamic>> get adminUsers => [
        {'id': 1, 'email': 'admin@demo.com', 'role': 'admin', 'user_type': 'user', 'lawyer_status': null, 'analyses_count': 0},
        {'id': 2, 'email': 'lawyer@demo.com', 'role': 'user', 'user_type': 'lawyer', 'lawyer_status': _apps[2]?.status ?? _statusOverrides[2] ?? 'pending', 'analyses_count': 3},
        {'id': 3, 'email': 'verified@demo.com', 'role': 'user', 'user_type': 'lawyer', 'lawyer_status': _statusOverrides[3] ?? 'approved', 'analyses_count': 8},
        {'id': 5, 'email': 'user@demo.com', 'role': 'user', 'user_type': 'user', 'lawyer_status': null, 'analyses_count': 5},
      ];

  // ── Social mock data ──────────────────────────────────────────────────────

  static Map<String, dynamic> profileOf(UserModel u) => {
        'user_id': u.id,
        'email': u.email,
        'display_name': nameOf(u),
        'title': titleOf(u),
        'company': u.isAdmin ? '' : 'Al-Rashid Legal Group',
        'location': 'Cairo, Egypt',
        'bio': u.isAdmin
            ? 'Platform administrator.'
            : 'Legal professional specialising in contract law and commercial disputes.',
        'avatar_url': null,
        'skills': u.isLawyerAccount
            ? ['Contract Law', 'Corporate Law', 'Dispute Resolution', 'Legal Analysis']
            : ['Legal Research', 'Contract Review'],
        'experience': <dynamic>[],
        'education': <dynamic>[],
        'stats': {'connections': 12, 'endorsements': 5},
        'user_type': u.userType,
        'lawyer_status': u.lawyerStatus,
        'is_verified_lawyer': u.isVerifiedLawyer,
      };

  static const List<Map<String, dynamic>> posts = [
    {
      'id': 1,
      'content': 'New Commercial Code amendments effective 2024: revised dispute resolution procedures, enhanced data protection clauses, and stricter compliance requirements for commercial contracts.',
      'category': 'Legal Insights',
      'created_at': '2024-01-15T08:30:00Z',
      'author_id': 3,
      'author_display_name': 'Sarah Al-Hassan',
      'author_title': 'Senior Legal Counsel',
      'author_avatar_url': null,
      'author_user_type': 'lawyer',
      'author_lawyer_status': 'approved',
      'author_is_verified_lawyer': true,
      'likes_count': 47,
      'comments_count': 12,
      'shares_count': 8,
      'is_liked': false,
      'tags': ['ContractLaw', 'LegalUpdates'],
      'image_url': null,
    },
    {
      'id': 2,
      'content': "Legato's AI contract analysis flagged 3 high-risk clauses in our M&A agreement that our team initially missed. Highly recommend for due diligence workflows.",
      'category': 'Technology',
      'created_at': '2024-01-14T14:20:00Z',
      'author_id': 5,
      'author_display_name': 'Karim Mansour',
      'author_title': 'Corporate Attorney',
      'author_avatar_url': null,
      'author_user_type': 'user',
      'author_lawyer_status': null,
      'author_is_verified_lawyer': false,
      'likes_count': 23,
      'comments_count': 5,
      'shares_count': 3,
      'is_liked': true,
      'tags': ['LegalTech', 'ContractReview'],
      'image_url': null,
    },
    {
      'id': 3,
      'content': 'Arbitration clauses in employment contracts are being challenged in multiple jurisdictions. Always review force majeure and dispute resolution provisions carefully.',
      'category': 'Employment Law',
      'created_at': '2024-01-13T11:00:00Z',
      'author_id': 3,
      'author_display_name': 'Sarah Al-Hassan',
      'author_title': 'Senior Legal Counsel',
      'author_avatar_url': null,
      'author_user_type': 'lawyer',
      'author_lawyer_status': 'approved',
      'author_is_verified_lawyer': true,
      'likes_count': 31,
      'comments_count': 8,
      'shares_count': 5,
      'is_liked': false,
      'tags': ['EmploymentLaw', 'Arbitration'],
      'image_url': null,
    },
  ];

  static const List<Map<String, dynamic>> networkSuggestions = [
    {
      'user_id': 3,
      'display_name': 'Sarah Al-Hassan',
      'name': 'Sarah Al-Hassan',
      'title': 'Senior Legal Counsel',
      'company': 'Al-Rashid Legal Group',
      'location': 'Cairo, Egypt',
      'avatar_url': null,
      'user_type': 'lawyer',
      'lawyer_status': 'approved',
      'is_verified_lawyer': true,
      'mutual_connections': 3,
    },
    {
      'user_id': 6,
      'display_name': 'Nour Abdel-Hamid',
      'name': 'Nour Abdel-Hamid',
      'title': 'Contract Specialist',
      'company': 'Egyptian Legal Partners',
      'location': 'Alexandria, Egypt',
      'avatar_url': null,
      'user_type': 'user',
      'lawyer_status': null,
      'is_verified_lawyer': false,
      'mutual_connections': 1,
    },
    {
      'user_id': 7,
      'display_name': 'Tarek Soliman',
      'name': 'Tarek Soliman',
      'title': 'Corporate Counsel',
      'company': 'Soliman & Associates',
      'location': 'Cairo, Egypt',
      'avatar_url': null,
      'user_type': 'lawyer',
      'lawyer_status': 'approved',
      'is_verified_lawyer': true,
      'mutual_connections': 2,
    },
  ];

  // ── Helpers ───────────────────────────────────────────────────────────────

  static String _emailFor(int id) => switch (id) {
        1 => 'admin@demo.com',
        2 => 'lawyer@demo.com',
        3 => 'verified@demo.com',
        4 => 'rejected@demo.com',
        _ => 'user@demo.com',
      };

  static UserModel _baseFor(int id) => switch (id) {
        1 => _adminBase,
        2 => _pendingBase,
        3 => _approvedBase,
        4 => _rejectedBase,
        _ => _userBase,
      };
}
