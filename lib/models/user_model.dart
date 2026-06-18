class UserModel {
  const UserModel({
    required this.id,
    required this.email,
    required this.role,
    this.userType = 'user',
    this.lawyerStatus,
  });

  final int id;
  final String email;
  final String role;
  final String userType;

  /// null for regular users; 'not_applied' | 'pending' | 'approved' | 'rejected' for lawyers.
  final String? lawyerStatus;

  bool get isAdmin => role.toLowerCase() == 'admin';

  /// True if the account was registered as a lawyer (any verification state).
  bool get isLawyerAccount => userType.toLowerCase() == 'lawyer';

  /// True only for lawyers whose application has been approved by an admin.
  bool get isVerifiedLawyer => isLawyerAccount && lawyerStatus == 'approved';

  /// True for lawyers who still need to submit or wait for verification.
  bool get needsLawyerVerification => isLawyerAccount && lawyerStatus != 'approved';

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      id: json['id'] as int,
      email: json['email'] as String,
      role: (json['role'] as String?) ?? 'user',
      userType: (json['user_type'] as String?) ?? 'user',
      lawyerStatus: json['lawyer_status'] as String?,
    );
  }
}
