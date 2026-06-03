class UserModel {
  const UserModel({required this.id, required this.email, required this.role});

  final int id;
  final String email;
  final String role;

  bool get isAdmin => role.toLowerCase() == 'admin';

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      id: json['id'] as int,
      email: json['email'] as String,
      role: (json['role'] as String?) ?? 'user',
    );
  }
}
