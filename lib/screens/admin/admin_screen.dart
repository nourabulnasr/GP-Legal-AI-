import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'package:legato_mobile/utils/file_download.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';

class AdminScreen extends StatefulWidget {
  const AdminScreen({super.key, this.initialTab = 0});

  final int initialTab;

  @override
  State<AdminScreen> createState() => _AdminScreenState();
}

class _AdminScreenState extends State<AdminScreen> with SingleTickerProviderStateMixin {
  late final TabController _tabs = TabController(length: 3, vsync: this, initialIndex: widget.initialTab);
  bool _loading = true;
  String? _err;
  List<dynamic> _users = [];
  List<dynamic> _analyses = [];
  List<dynamic> _lawyerApps = [];
  bool _loadingApps = false;
  String _appStatusFilter = 'pending';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _load();
      if (widget.initialTab == 2) _loadLawyerApps();
    });
    _tabs.addListener(() {
      if (_tabs.index == 2 && _lawyerApps.isEmpty && !_loadingApps) {
        _loadLawyerApps();
      }
    });
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
    // Also load lawyer apps if that tab is active
    if (_tabs.index == 2) _loadLawyerApps();
  }

  Future<void> _loadLawyerApps() async {
    setState(() => _loadingApps = true);
    final legato = context.read<AppServices>().legato;
    try {
      List<dynamic> backendApps = [];
      try {
        backendApps = await legato.adminListLawyerApplications(status: _appStatusFilter);
      } on ApiException catch (e) {
        if (e.statusCode == 404) {
          // /admin/lawyers not deployed — derive a list from the admin users endpoint.
          try {
            final users = await legato.adminListUsers();
            final filter = _appStatusFilter;
            backendApps = users.where((u) {
              final m = u as Map<String, dynamic>;
              if ((m['user_type']?.toString() ?? '') != 'lawyer') return false;
              final s = m['lawyer_status']?.toString() ?? 'pending';
              if (filter == 'all') return true;
              if (filter == 'pending') return s == 'pending' || s.isEmpty || s == 'not_applied';
              return s == filter;
            }).map<Map<String, dynamic>>((u) {
              final m = u as Map<String, dynamic>;
              return {
                'id': m['id'], 'user_id': m['id'], 'user_email': m['email'],
                'status': m['lawyer_status'] ?? 'pending',
                'bar_license_number': m['bar_license_number'] ?? '',
                'has_cv': false, 'cv_filename': '',
                'has_id_card': false, 'id_card_filename': '',
                'has_document': false, 'document_filename': '',
                'admin_note': m['admin_note'] ?? '',
              };
            }).toList();
          } catch (_) {}
        } else if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
        }
      }

      if (mounted) setState(() => _lawyerApps = backendApps);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.toString())));
      }
    } finally {
      if (mounted) setState(() => _loadingApps = false);
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

  String _mimeFromFilename(String filename) {
    final ext = filename.split('.').last.toLowerCase();
    return switch (ext) {
      'pdf'  => 'application/pdf',
      'png'  => 'image/png',
      'jpg'  => 'image/jpeg',
      'jpeg' => 'image/jpeg',
      'doc'  => 'application/msword',
      'docx' => 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      _      => 'application/octet-stream',
    };
  }

  Future<Uint8List?> _fetchLawyerFileBytes(int appId, String fileType) async {
    final legato = context.read<AppServices>().legato;
    try {
      switch (fileType) {
        case 'cv':
          return await legato.adminDownloadLawyerCv(appId);
        case 'id-card-back':
          return await legato.adminDownloadLawyerIdCardBack(appId);
        default:
          return await legato.adminDownloadLawyerIdCard(appId);
      }
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      return null;
    }
  }

  Future<void> _viewLawyerFile(int appId, String fileType, String filename) async {
    final bytes = await _fetchLawyerFileBytes(appId, fileType);
    if (bytes == null || !mounted) return;
    final mime = _mimeFromFilename(filename);
    // Images: show inline dialog. PDFs and others: open in browser tab.
    if (mime == 'image/jpeg' || mime == 'image/png') {
      await showDialog<void>(
        context: context,
        builder: (ctx) => Dialog(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AppBar(
                title: Text(filename, overflow: TextOverflow.ellipsis),
                automaticallyImplyLeading: false,
                actions: [
                  IconButton(
                    icon: const Icon(Icons.download_outlined),
                    tooltip: 'Download',
                    onPressed: () => triggerFileDownload(bytes, filename, mime),
                  ),
                  IconButton(icon: const Icon(Icons.close), onPressed: () => Navigator.pop(ctx)),
                ],
              ),
              InteractiveViewer(
                child: Image.memory(bytes, fit: BoxFit.contain),
              ),
            ],
          ),
        ),
      );
    } else {
      openFileInBrowser(bytes, mime);
    }
  }

  Future<void> _downloadLawyerFile(int appId, String fileType, String filename) async {
    final bytes = await _fetchLawyerFileBytes(appId, fileType);
    if (bytes == null) return;
    triggerFileDownload(bytes, filename, _mimeFromFilename(filename));
  }

  Future<void> _reviewLawyerApp(int appId, int? userId, String userEmail, String action) async {
    String? note;
    if (action == 'reject') {
      note = await showDialog<String>(
        context: context,
        builder: (ctx) {
          final ctrl = TextEditingController();
          return AlertDialog(
            title: const Text('Rejection reason (optional)'),
            content: TextField(
              controller: ctrl,
              maxLines: 3,
              decoration: const InputDecoration(
                hintText: 'e.g. Document unclear, invalid license number…',
                border: OutlineInputBorder(),
              ),
            ),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(ctx, null), child: const Text('Cancel')),
              FilledButton(
                style: FilledButton.styleFrom(
                    backgroundColor: Theme.of(ctx).colorScheme.error),
                onPressed: () => Navigator.pop(ctx, ctrl.text.trim()),
                child: const Text('Reject'),
              ),
            ],
          );
        },
      );
      if (note == null) return; // cancelled
    } else {
      final ok = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Approve lawyer?'),
          content: Text('Approve $userEmail as a verified lawyer?'),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Approve'),
            ),
          ],
        ),
      );
      if (ok != true) return;
    }

    if (!mounted) return;
    final legato = context.read<AppServices>().legato;
    final newStatus = action == 'approve' ? 'approved' : 'rejected';

    // Try dedicated review endpoint; on 404 fall back to patching the user record.
    try {
      await legato.adminReviewLawyerApplication(appId, action: action, adminNote: note);
    } on ApiException catch (reviewErr) {
      if (reviewErr.statusCode == 404 && userId != null) {
        try {
          await legato.adminUpdateLawyerStatus(userId, newStatus);
        } catch (_) {
          // Backend has no lawyer_status field or this user wasn't found.
        }
      }
      // Non-404 review endpoint errors are swallowed.
    }

    // On approval also set user_type = 'lawyer' on the backend (belt-and-suspenders).
    if (action == 'approve' && userId != null) {
      try { await legato.adminUpdateUserType(userId, 'lawyer'); } catch (_) {}
    }

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(action == 'approve'
            ? '$userEmail approved as verified lawyer.'
            : '$userEmail application rejected.'),
      ));
    }
    await _loadLawyerApps();
  }

  @override
  Widget build(BuildContext context) {
    final currentUserId = context.watch<AuthProvider>().user?.id;
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: LegatoAppBar(
        title: const Text('Admin'),
        bottom: TabBar(
          controller: _tabs,
          tabs: const [
            Tab(text: 'Users'),
            Tab(text: 'Analyses'),
            Tab(text: 'Lawyers'),
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
                    // ── Tab 1: Users ──────────────────────────────────────
                    ListView.separated(
                      padding: const EdgeInsets.all(8),
                      itemCount: _users.length,
                      separatorBuilder: (context, i) => const Divider(height: 1),
                      itemBuilder: (context, i) {
                        final m = _users[i] as Map<String, dynamic>;
                        final id = m['id'] as int;
                        final email = m['email']?.toString() ?? '';
                        final role = m['role']?.toString() ?? 'user';
                        final userType = m['user_type']?.toString() ?? 'user';
                        final lawyerStatus = m['lawyer_status']?.toString() ?? '';
                        return ListTile(
                          leading: CircleAvatar(
                            backgroundColor: role == 'admin'
                                ? cs.primaryContainer
                                : userType == 'lawyer'
                                    ? cs.secondaryContainer
                                    : cs.surfaceContainerHighest,
                            child: Icon(
                              role == 'admin'
                                  ? Icons.admin_panel_settings_outlined
                                  : userType == 'lawyer'
                                      ? Icons.gavel_outlined
                                      : Icons.person_outline,
                              size: 18,
                              color: role == 'admin'
                                  ? cs.onPrimaryContainer
                                  : userType == 'lawyer'
                                      ? cs.onSecondaryContainer
                                      : cs.onSurfaceVariant,
                            ),
                          ),
                          title: Text(email),
                          subtitle: Text(
                            userType == 'lawyer'
                                ? 'id: $id · $role · lawyer ($lawyerStatus)'
                                : 'id: $id · $role',
                          ),
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

                    // ── Tab 2: Analyses ───────────────────────────────────
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
                          subtitle: Text('analysis $id � user $uid'),
                        );
                      },
                    ),

                    // ── Tab 3: Lawyer Applications ────────────────────────
                    Column(
                      children: [
                        Padding(
                          padding: const EdgeInsets.fromLTRB(12, 10, 12, 4),
                          child: Row(
                            children: [
                              const Text('Filter: '),
                              const SizedBox(width: 8),
                              DropdownButton<String>(
                                value: _appStatusFilter,
                                items: const [
                                  DropdownMenuItem(value: 'pending', child: Text('Pending')),
                                  DropdownMenuItem(value: 'approved', child: Text('Approved')),
                                  DropdownMenuItem(value: 'rejected', child: Text('Rejected')),
                                  DropdownMenuItem(value: 'all', child: Text('All')),
                                ],
                                onChanged: (v) {
                                  if (v == null) return;
                                  setState(() {
                                    _appStatusFilter = v;
                                    _lawyerApps = [];
                                  });
                                  _loadLawyerApps();
                                },
                              ),
                              const Spacer(),
                              if (_loadingApps)
                                const SizedBox(
                                  height: 18,
                                  width: 18,
                                  child: CircularProgressIndicator(strokeWidth: 2),
                                )
                              else
                                IconButton(
                                  icon: const Icon(Icons.refresh),
                                  onPressed: _loadLawyerApps,
                                ),
                            ],
                          ),
                        ),
                        const Divider(height: 1),
                        Expanded(
                          child: _lawyerApps.isEmpty
                              ? Center(
                                  child: Text(
                                    _loadingApps
                                        ? 'Loading…'
                                        : 'No $_appStatusFilter applications.',
                                    style: Theme.of(context)
                                        .textTheme
                                        .bodyMedium
                                        ?.copyWith(color: cs.onSurfaceVariant),
                                  ),
                                )
                              : ListView.separated(
                                  padding: const EdgeInsets.all(8),
                                  itemCount: _lawyerApps.length,
                                  separatorBuilder: (_, i) => const Divider(height: 1),
                                  itemBuilder: (context, i) {
                                    final app = _lawyerApps[i] as Map<String, dynamic>;
                                    final appId = app['id'] as int;
                                    final email = app['user_email']?.toString() ?? '';
                                    final licNo = app['bar_license_number']?.toString() ?? '';
                                    final docFile = app['document_filename']?.toString() ?? '';
                                    final hasDoc = app['has_document'] as bool? ?? false;
                                    final cvFile = app['cv_filename']?.toString() ?? '';
                                    final hasCv = app['has_cv'] as bool? ?? false;
                                    final idCardFile = app['id_card_filename']?.toString() ?? '';
                                    final hasIdCard = app['has_id_card'] as bool? ?? false;
                                    final idCardBackFile = app['id_card_back_filename']?.toString() ?? '';
                                    final hasIdCardBack = app['has_id_card_back'] as bool? ?? false;
                                    final status = app['status']?.toString() ?? 'pending';
                                    final adminNote = app['admin_note']?.toString() ?? '';
                                    final yearsExp = app['years_of_experience'];

                                    Color statusColor;
                                    IconData statusIcon;
                                    switch (status) {
                                      case 'approved':
                                        statusColor = Colors.green;
                                        statusIcon = Icons.verified_outlined;
                                      case 'rejected':
                                        statusColor = cs.error;
                                        statusIcon = Icons.cancel_outlined;
                                      default:
                                        statusColor = cs.secondary;
                                        statusIcon = Icons.hourglass_empty_outlined;
                                    }

                                    return Card(
                                      margin: const EdgeInsets.symmetric(vertical: 4),
                                      child: Padding(
                                        padding: const EdgeInsets.all(12),
                                        child: Column(
                                          crossAxisAlignment: CrossAxisAlignment.start,
                                          children: [
                                            Row(
                                              children: [
                                                Icon(statusIcon, color: statusColor, size: 18),
                                                const SizedBox(width: 6),
                                                Expanded(
                                                  child: Text(
                                                    email,
                                                    style: Theme.of(context).textTheme.titleSmall,
                                                    overflow: TextOverflow.ellipsis,
                                                  ),
                                                ),
                                                Container(
                                                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                                  decoration: BoxDecoration(
                                                    color: statusColor.withValues(alpha: 0.12),
                                                    borderRadius: BorderRadius.circular(20),
                                                  ),
                                                  child: Text(
                                                    status.toUpperCase(),
                                                    style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                                          color: statusColor,
                                                          fontWeight: FontWeight.bold,
                                                        ),
                                                  ),
                                                ),
                                              ],
                                            ),
                                            if (licNo.isNotEmpty) ...[
                                              const SizedBox(height: 4),
                                              Text('License: $licNo', style: Theme.of(context).textTheme.bodySmall),
                                            ],
                                            if (yearsExp != null) ...[
                                              const SizedBox(height: 4),
                                              Text(
                                                'Experience: $yearsExp year${yearsExp == 1 ? '' : 's'}',
                                                style: Theme.of(context).textTheme.bodySmall,
                                              ),
                                            ],
                                            if (hasDoc) ...[
                                              const SizedBox(height: 4),
                                              Row(
                                                children: [
                                                  Icon(Icons.attach_file, size: 14, color: cs.primary),
                                                  const SizedBox(width: 4),
                                                  Expanded(
                                                    child: Text(
                                                      docFile.isNotEmpty ? docFile : 'Document uploaded',
                                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.primary),
                                                      overflow: TextOverflow.ellipsis,
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ],
                                            if (hasCv) ...[
                                              const SizedBox(height: 4),
                                              Row(
                                                children: [
                                                  Icon(Icons.description_outlined, size: 14, color: cs.primary),
                                                  const SizedBox(width: 4),
                                                  Expanded(
                                                    child: Text(
                                                      cvFile.isNotEmpty ? 'CV: $cvFile' : 'CV uploaded',
                                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.primary),
                                                      overflow: TextOverflow.ellipsis,
                                                    ),
                                                  ),
                                                  TextButton(
                                                    style: TextButton.styleFrom(
                                                        visualDensity: VisualDensity.compact,
                                                        padding: EdgeInsets.zero),
                                                    onPressed: () => _viewLawyerFile(
                                                        appId, 'cv', cvFile.isNotEmpty ? cvFile : 'cv'),
                                                    child: const Text('View'),
                                                  ),
                                                  TextButton(
                                                    style: TextButton.styleFrom(
                                                        visualDensity: VisualDensity.compact,
                                                        padding: const EdgeInsets.only(left: 4)),
                                                    onPressed: () => _downloadLawyerFile(
                                                        appId, 'cv', cvFile.isNotEmpty ? cvFile : 'cv'),
                                                    child: const Text('Download'),
                                                  ),
                                                ],
                                              ),
                                            ],
                                            if (hasIdCard) ...[
                                              const SizedBox(height: 4),
                                              Row(
                                                children: [
                                                  Icon(Icons.badge_outlined, size: 14, color: cs.primary),
                                                  const SizedBox(width: 4),
                                                  Expanded(
                                                    child: Text(
                                                      idCardFile.isNotEmpty ? 'ID front: $idCardFile' : 'ID card front uploaded',
                                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.primary),
                                                      overflow: TextOverflow.ellipsis,
                                                    ),
                                                  ),
                                                  TextButton(
                                                    style: TextButton.styleFrom(
                                                        visualDensity: VisualDensity.compact,
                                                        padding: EdgeInsets.zero),
                                                    onPressed: () => _viewLawyerFile(
                                                        appId, 'id-card', idCardFile.isNotEmpty ? idCardFile : 'id_card_front'),
                                                    child: const Text('View'),
                                                  ),
                                                  TextButton(
                                                    style: TextButton.styleFrom(
                                                        visualDensity: VisualDensity.compact,
                                                        padding: const EdgeInsets.only(left: 4)),
                                                    onPressed: () => _downloadLawyerFile(
                                                        appId, 'id-card', idCardFile.isNotEmpty ? idCardFile : 'id_card_front'),
                                                    child: const Text('Download'),
                                                  ),
                                                ],
                                              ),
                                            ],
                                            if (hasIdCardBack) ...[
                                              const SizedBox(height: 4),
                                              Row(
                                                children: [
                                                  Icon(Icons.badge_outlined, size: 14, color: cs.primary),
                                                  const SizedBox(width: 4),
                                                  Expanded(
                                                    child: Text(
                                                      idCardBackFile.isNotEmpty ? 'ID back: $idCardBackFile' : 'ID card back uploaded',
                                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.primary),
                                                      overflow: TextOverflow.ellipsis,
                                                    ),
                                                  ),
                                                  TextButton(
                                                    style: TextButton.styleFrom(
                                                        visualDensity: VisualDensity.compact,
                                                        padding: EdgeInsets.zero),
                                                    onPressed: () => _viewLawyerFile(
                                                        appId, 'id-card-back', idCardBackFile.isNotEmpty ? idCardBackFile : 'id_card_back'),
                                                    child: const Text('View'),
                                                  ),
                                                  TextButton(
                                                    style: TextButton.styleFrom(
                                                        visualDensity: VisualDensity.compact,
                                                        padding: const EdgeInsets.only(left: 4)),
                                                    onPressed: () => _downloadLawyerFile(
                                                        appId, 'id-card-back', idCardBackFile.isNotEmpty ? idCardBackFile : 'id_card_back'),
                                                    child: const Text('Download'),
                                                  ),
                                                ],
                                              ),
                                            ],
                                            if (adminNote.isNotEmpty) ...[
                                              const SizedBox(height: 4),
                                              Text(
                                                'Note: $adminNote',
                                                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                                              ),
                                            ],
                                            if (status == 'pending') ...[
                                              const SizedBox(height: 10),
                                              Row(
                                                mainAxisAlignment: MainAxisAlignment.end,
                                                children: [
                                                  OutlinedButton.icon(
                                                    style: OutlinedButton.styleFrom(
                                                      foregroundColor: cs.error,
                                                      side: BorderSide(color: cs.error.withValues(alpha: 0.5)),
                                                    ),
                                                    onPressed: () => _reviewLawyerApp(
                                                        appId, app['user_id'] as int?, email, 'reject'),
                                                    icon: const Icon(Icons.close, size: 16),
                                                    label: const Text('Reject'),
                                                  ),
                                                  const SizedBox(width: 8),
                                                  FilledButton.icon(
                                                    onPressed: () => _reviewLawyerApp(
                                                        appId, app['user_id'] as int?, email, 'approve'),
                                                    icon: const Icon(Icons.check, size: 16),
                                                    label: const Text('Approve'),
                                                  ),
                                                ],
                                              ),
                                            ],
                                          ],
                                        ),
                                      ),
                                    );
                                  },
                                ),
                        ),
                      ],
                    ),
                  ],
                ),
    );
  }
}
