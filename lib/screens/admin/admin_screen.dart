import 'package:flutter/material.dart';

import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';

class AdminScreen extends StatefulWidget {
  const AdminScreen({super.key});

  @override
  State<AdminScreen> createState() => _AdminScreenState();
}

class _AdminScreenState extends State<AdminScreen> with SingleTickerProviderStateMixin {
  late final TabController _tabs = TabController(length: 2, vsync: this);
  bool _loading = true;
  String? _err;
  List<dynamic> _users = [];
  List<dynamic> _analyses = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    final api = context.read<AppServices>().legato;
    try {
      final u = await api.adminListUsers();
      final a = await api.adminListAll();
      setState(() {
        _users = u;
        _analyses = a;
      });
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _setRole(int userId, String role) async {
    try {
      await context.read<AppServices>().legato.adminUpdateUserRole(userId, role);
      await _load();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Role ? $role')));
      }
    } on ApiException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      }
    }
  }

  Future<void> _deleteUser(int userId, String email) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete user?'),
        content: Text(
          'Permanently delete $email and all their analyses, posts, and profile data? This cannot be undone.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Theme.of(ctx).colorScheme.error),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok != true || !mounted) return;
    try {
      await context.read<AppServices>().legato.adminDeleteUser(userId);
      await _load();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Deleted $email')));
      }
    } on ApiException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final currentUserId = context.watch<AuthProvider>().user?.id;
    return Scaffold(
      appBar: LegatoAppBar(
        title: const Text('Admin'),
        bottom: TabBar(
          controller: _tabs,
          tabs: const [
            Tab(text: 'Users'),
            Tab(text: 'All analyses'),
          ],
        ),
        actions: [
          IconButton(onPressed: _loading ? null : _load, icon: const Icon(Icons.refresh)),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _err != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(_err!, textAlign: TextAlign.center),
                        const SizedBox(height: 12),
                        FilledButton(onPressed: _load, child: const Text('Retry')),
                      ],
                    ),
                  ),
                )
              : TabBarView(
                  controller: _tabs,
                  children: [
                    ListView.separated(
                      padding: const EdgeInsets.all(8),
                      itemCount: _users.length,
                      separatorBuilder: (context, i) => const Divider(height: 1),
                      itemBuilder: (context, i) {
                        final m = _users[i] as Map<String, dynamic>;
                        final id = m['id'] as int;
                        final email = m['email']?.toString() ?? '';
                        final role = m['role']?.toString() ?? 'user';
                        return ListTile(
                          title: Text(email),
                          subtitle: Text('id: $id  $role'),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              if (id != currentUserId)
                                IconButton(
                                  tooltip: 'Delete user',
                                  icon: Icon(Icons.delete_outline, color: Theme.of(context).colorScheme.error),
                                  onPressed: () => _deleteUser(id, email),
                                ),
                              if (role == 'admin')
                                TextButton(
                                  onPressed: id == currentUserId ? null : () => _setRole(id, 'user'),
                                  child: const Text('Make user'),
                                )
                              else
                                TextButton(
                                  onPressed: () => _setRole(id, 'admin'),
                                  child: const Text('Make admin'),
                                ),
                            ],
                          ),
                        );
                      },
                    ),
                    ListView.separated(
                      padding: const EdgeInsets.all(8),
                      itemCount: _analyses.length,
                      separatorBuilder: (context, i) => const Divider(height: 1),
                      itemBuilder: (context, i) {
                        final m = _analyses[i] as Map<String, dynamic>;
                        final id = m['id'];
                        final fn = m['filename']?.toString() ?? '';
                        final uid = m['user_id']?.toString() ?? '';
                        return ListTile(
                          title: Text(fn),
                          subtitle: Text('analysis $id · user $uid'),
                        );
                      },
                    ),
                  ],
                ),
    );
  }
}
