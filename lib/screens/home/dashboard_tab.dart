import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/utils/local_lawyer_queue.dart';
import 'package:legato_mobile/providers/user_profile_provider.dart';
import 'package:legato_mobile/screens/analyze/analyze_screen.dart';
import 'package:legato_mobile/screens/features/features_hub_screen.dart';
import 'package:legato_mobile/screens/translate/translate_contract_screen.dart';
import 'package:legato_mobile/screens/chat/chat_hub_screen.dart';
import 'package:legato_mobile/screens/history/history_screen.dart';
import 'package:legato_mobile/screens/admin/admin_screen.dart';
import 'package:legato_mobile/screens/lawyer/lawyer_application_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/user_avatar.dart';

class DashboardTab extends StatefulWidget {
  const DashboardTab({super.key});

  @override
  State<DashboardTab> createState() => DashboardTabState();
}

class DashboardTabState extends State<DashboardTab> {
  void refresh() {
    final auth = context.read<AuthProvider>();
    context.read<UserProfileProvider>().refresh(
          userId: auth.user?.id,
          email: auth.user?.email ?? '',
        );
    _checkConnection();
    _loadPendingAppsIfAdmin();
    _checkLocalEntry();
  }

  bool? _connected;
  Timer? _connectionTimer;
  int _pendingLawyerAppsCount = 0;
  bool _hasLocalEntry = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _checkConnection();
      _loadPendingAppsIfAdmin();
      _checkLocalEntry();
    });
    _connectionTimer = Timer.periodic(const Duration(seconds: 45), (_) => _checkConnection());
  }

  Future<void> _loadPendingAppsIfAdmin() async {
    final user = context.read<AuthProvider>().user;
    if (user == null || !user.isAdmin) return;
    final legato = context.read<AppServices>().legato;
    try {
      final apps = await legato.adminListLawyerApplications(status: 'pending');
      final localPending = await LocalLawyerQueue.getPending();
      if (mounted) setState(() => _pendingLawyerAppsCount = apps.length + localPending.length);
    } on ApiException catch (e) {
      if (e.statusCode == 404) {
        // /admin/lawyers not deployed yet — fall back to the users list and
        // count lawyer-type accounts whose verification is still pending.
        try {
          final users = await legato.adminListUsers();
          final count = users.where((u) {
            final m = u as Map<String, dynamic>;
            final t = m['user_type']?.toString() ?? '';
            final s = m['lawyer_status']?.toString() ?? '';
            return t == 'lawyer' && (s == 'pending' || s.isEmpty || s == 'not_applied');
          }).length;
          final localPending = await LocalLawyerQueue.getPending();
          if (mounted) setState(() => _pendingLawyerAppsCount = count + localPending.length);
        } catch (_) {}
      }
    } catch (_) {}
  }

  Future<void> _checkLocalEntry() async {
    final userId = context.read<AuthProvider>().user?.id;
    if (userId == null) return;
    try {
      final all = await LocalLawyerQueue.getAll();
      final has = all.any((m) => (m['user_id'] as int?) == userId);
      if (mounted) setState(() => _hasLocalEntry = has);
    } catch (_) {}
  }

  @override
  void dispose() {
    _connectionTimer?.cancel();
    super.dispose();
  }

  Future<void> _checkConnection() async {
    try {
      await context.read<AppServices>().api.getHealthRaw();
      if (mounted) setState(() => _connected = true);
    } catch (_) {
      if (mounted) setState(() => _connected = false);
    }
  }

  Widget _connectionBadge() {
    final ok = _connected == true;
    final color = ok ? Colors.greenAccent : Colors.redAccent;
    final label = ok ? 'Connected' : 'Disconnected';
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 9,
          height: 9,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(
          label,
          style: TextStyle(color: Colors.white.withValues(alpha: 0.95), fontSize: 12, fontWeight: FontWeight.w500),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final profile = context.watch<UserProfileProvider>();
    final user = auth.user;
    final rawEmail = user?.email ?? 'Guest';
    final displayName = profile.displayName.isNotEmpty
        ? profile.displayName
        : (rawEmail.contains('@') ? rawEmail.split('@').first : rawEmail);

    return ColoredBox(
      color: Theme.of(context).scaffoldBackgroundColor,
      child: SafeArea(
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Container(
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        colors: [Color(0xFF1B1F23), LegatoLinkedInTheme.navActiveGold],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.vertical(bottom: Radius.circular(12)),
                    ),
                    padding: const EdgeInsets.fromLTRB(20, 12, 20, 28),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Align(alignment: Alignment.topRight, child: _connectionBadge()),
                        const SizedBox(height: 4),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            UserAvatar(
                              radius: 36,
                              imageUrl: profile.avatarUrl,
                              name: displayName,
                              backgroundColor: Colors.white,
                            ),
                            const SizedBox(width: 16),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Welcome back',
                                    style: TextStyle(color: Colors.white.withValues(alpha: 0.9), fontSize: 13),
                                  ),
                                  const SizedBox(height: 4),
                                  Row(
                                    children: [
                                      Flexible(
                                        child: Text(
                                          displayName,
                                          overflow: TextOverflow.ellipsis,
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 18,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ),
                                      if (user?.isVerifiedLawyer == true) ...[
                                        const SizedBox(width: 6),
                                        const Tooltip(
                                          message: 'Verified Lawyer',
                                          child: Icon(Icons.verified, size: 16, color: Color(0xFF0A66C2)),
                                        ),
                                      ],
                                    ],
                                  ),
                                  if (user != null) ...[
                                    const SizedBox(height: 4),
                                    Text(
                                      user.isVerifiedLawyer
                                          ? 'VERIFIED LAWYER'
                                          : user.role.toUpperCase(),
                                      style: TextStyle(color: Colors.white.withValues(alpha: 0.85), fontSize: 12),
                                    ),
                                  ],
                                ],
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  Transform.translate(
                    offset: const Offset(0, -18),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      child: Column(
                        children: [
                          if (user != null && user.isAdmin && _pendingLawyerAppsCount > 0)
                            Padding(
                              padding: const EdgeInsets.only(top: 18),
                              child: _AdminPendingLawyersBanner(
                count: _pendingLawyerAppsCount,
                onReturn: _loadPendingAppsIfAdmin,
              ),
                            ),
                          // Show banner for any user who has an active lawyer
                          // application (pending or rejected) or who is a
                          // lawyer-type account not yet verified.
                          if (user != null &&
                              !user.isAdmin &&
                              !user.isVerifiedLawyer &&
                              (user.isLawyerAccount ||
                                  _hasLocalEntry ||
                                  (user.lawyerStatus != null &&
                                      user.lawyerStatus != 'not_applied')))
                            Padding(
                              padding: const EdgeInsets.only(top: 18),
                              child: _LawyerStatusBanner(lawyerStatus: user.lawyerStatus),
                            ),
                          Card(
                            child: Padding(
                              padding: const EdgeInsets.all(16),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.stretch,
                                children: [
                                  Text(
                                    'Workspace',
                                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                          fontWeight: FontWeight.w600,
                                        ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    'Analyze contracts, review history, and use legal tools — same API as the web app.',
                                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                          color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                                        ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
              sliver: SliverList.list(
                children: [
                  Text(
                    'Quick actions',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          fontWeight: FontWeight.w600,
                        ),
                  ),
                  const SizedBox(height: 8),
                  _ActionCard(
                    icon: Icons.upload_file_outlined,
                    title: 'Analyze contract',
                    subtitle: 'Upload PDF / DOCX — OCR + rules + RAG + LFM',
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(builder: (_) => const AnalyzeScreen()),
                    ),
                  ),
                  const SizedBox(height: 8),
                  _ActionCard(
                    icon: Icons.translate,
                    title: 'Translate contract',
                    subtitle: 'OCR + automatic translation (no engine picker)',
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(builder: (_) => const TranslateContractScreen()),
                    ),
                  ),
                  const SizedBox(height: 8),
                  _ActionCard(
                    icon: Icons.history,
                    title: 'Analysis history',
                    subtitle: 'View and revisit saved contract analyses',
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(builder: (_) => const HistoryScreen()),
                    ),
                  ),
                  const SizedBox(height: 8),
                  _ActionCard(
                    icon: Icons.smart_toy_outlined,
                    title: 'AI Chat',
                    subtitle: 'General assistant (Gemini) · Document chat (LFM)',
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(builder: (_) => const ChatHubScreen()),
                    ),
                  ),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: LegatoLinkedInTheme.navActiveGold,
                      foregroundColor: const Color(0xFF1B1F23),
                    ),
                    onPressed: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(builder: (_) => const FeaturesHubScreen()),
                    ),
                    icon: const Icon(Icons.apps_outlined),
                    label: Text(FeaturesHubScreen.openAllToolsLabel),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LawyerStatusBanner extends StatefulWidget {
  const _LawyerStatusBanner({required this.lawyerStatus});

  final String? lawyerStatus;

  @override
  State<_LawyerStatusBanner> createState() => _LawyerStatusBannerState();
}

class _LawyerStatusBannerState extends State<_LawyerStatusBanner> {
  Map<String, dynamic>? _data;

  @override
  void initState() {
    super.initState();
    _fetchStatus();
  }

  Future<void> _fetchStatus() async {
    final legato = context.read<AppServices>().legato;
    final authProv = context.read<AuthProvider>();
    try {
      final data = await legato.lawyerStatus();
      final backendStatus = data['status'] as String? ?? 'not_applied';

      // Backend has no record for this user — check whether the local queue has
      // a more up-to-date status (application submitted when /lawyer/apply was 404,
      // or approved/rejected locally by the admin).
      if (backendStatus == 'not_applied') {
        final userId = authProv.user?.id;
        if (userId != null) {
          try {
            final all = await LocalLawyerQueue.getAll();
            Map<String, dynamic>? entry;
            for (final m in all) {
              if ((m['user_id'] as int?) == userId) { entry = m; break; }
            }
            if (entry != null) {
              final localStatus = entry['status']?.toString() ?? 'pending';
              if (localStatus == 'approved') {
                // Refresh first so isVerifiedLawyer becomes true; banner will
                // be removed from the tree (mounted becomes false).
                await authProv.refreshUser();
                if (!mounted) return;
              }
              setState(() => _data = entry);
              return;
            }
          } catch (_) {}
        }
      }

      if (!mounted) return;
      setState(() => _data = data);
      if (backendStatus == 'approved') {
        await authProv.refreshUser();
      }
    } on ApiException catch (e) {
      if (e.statusCode == 404) {
        // /lawyer/status not deployed — check local queue first, then user model.
        if (!mounted) return;
        String lawyerStatus = authProv.user?.lawyerStatus ?? 'pending';
        Map<String, dynamic>? queueData;
        try {
          final userId = authProv.user?.id;
          if (userId != null) {
            final all = await LocalLawyerQueue.getAll();
            final entry = all.cast<Map<String, dynamic>?>().firstWhere(
              (m) => (m?['user_id'] as int?) == userId,
              orElse: () => null,
            );
            if (entry != null) {
              lawyerStatus = entry['status']?.toString() ?? lawyerStatus;
              queueData = entry;
            }
          }
        } catch (_) {}
        if (!mounted) return;
        setState(() => _data = queueData ?? {'status': lawyerStatus});
        if (lawyerStatus == 'approved') {
          await authProv.refreshUser();
        }
      }
    } catch (_) {}
  }

  Future<void> _openApplication() async {
    await Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const LawyerApplicationScreen()),
    );
    if (mounted) _fetchStatus();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final status = (_data?['status'] as String?) ?? widget.lawyerStatus ?? 'not_applied';

    final IconData icon;
    final Color color;
    final String title;

    switch (status) {
      case 'pending':
        icon = Icons.hourglass_empty_outlined;
        color = cs.secondary;
        title = 'Application Under Review';
      case 'rejected':
        icon = Icons.cancel_outlined;
        color = cs.error;
        title = 'Application Rejected';
      case 'approved':
        icon = Icons.verified_outlined;
        color = Colors.green;
        title = 'Verified Lawyer';
      default:
        icon = Icons.gavel_outlined;
        color = cs.primary;
        title = 'Complete Lawyer Verification';
    }

    final cvFile = _data?['cv_filename']?.toString() ?? '';
    final idFile = _data?['id_card_filename']?.toString() ?? '';
    final licNo  = _data?['bar_license_number']?.toString() ?? '';
    final adminNote = _data?['admin_note']?.toString() ?? '';
    final hasDocs = cvFile.isNotEmpty || idFile.isNotEmpty;

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withValues(alpha: 0.35)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header row ───────────────────────────────────────────────
            Row(
              children: [
                Icon(icon, color: color, size: 22),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    title,
                    style: Theme.of(context)
                        .textTheme
                        .titleSmall
                        ?.copyWith(color: color, fontWeight: FontWeight.bold),
                  ),
                ),
                // Status chip
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    status.toUpperCase(),
                    style: Theme.of(context)
                        .textTheme
                        .labelSmall
                        ?.copyWith(color: color, fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),

            // ── Submitted documents ───────────────────────────────────────
            if (hasDocs) ...[
              const SizedBox(height: 10),
              const Divider(height: 1),
              const SizedBox(height: 8),
              Text(
                'Submitted documents',
                style: Theme.of(context)
                    .textTheme
                    .labelSmall
                    ?.copyWith(color: cs.onSurfaceVariant),
              ),
              const SizedBox(height: 6),
              if (cvFile.isNotEmpty)
                _DocRow(icon: Icons.description_outlined, label: 'CV / Resume', filename: cvFile),
              if (idFile.isNotEmpty)
                _DocRow(icon: Icons.badge_outlined, label: 'National ID Card', filename: idFile),
              if (licNo.isNotEmpty)
                _DocRow(icon: Icons.numbers_outlined, label: 'Bar License', filename: licNo),
            ] else if (status == 'not_applied') ...[
              const SizedBox(height: 6),
              Text(
                'Upload your CV and national ID card to get verified as a lawyer.',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],

            // ── Admin note (rejection reason) ─────────────────────────────
            if (status == 'rejected' && adminNote.isNotEmpty) ...[
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: cs.error.withValues(alpha: 0.06),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: cs.error.withValues(alpha: 0.2)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.info_outline, size: 14, color: cs.error),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        adminNote,
                        style: Theme.of(context)
                            .textTheme
                            .bodySmall
                            ?.copyWith(color: cs.error),
                      ),
                    ),
                  ],
                ),
              ),
            ],

            // ── Action button ──────────────────────────────────────────────
            if (status != 'pending' && status != 'approved') ...[
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: FilledButton.tonal(
                  style: FilledButton.styleFrom(
                    backgroundColor: color.withValues(alpha: 0.15),
                    foregroundColor: color,
                    padding: const EdgeInsets.symmetric(vertical: 10),
                  ),
                  onPressed: _openApplication,
                  child: Text(status == 'rejected' ? 'Resubmit Application' : 'Apply Now'),
                ),
              ),
            ] else ...[
              const SizedBox(height: 8),
              Text(
                'Check the Alerts tab for updates.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: cs.onSurfaceVariant,
                    ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _DocRow extends StatelessWidget {
  const _DocRow({required this.icon, required this.label, required this.filename});

  final IconData icon;
  final String label;
  final String filename;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          Icon(icon, size: 14, color: cs.primary),
          const SizedBox(width: 6),
          Text(
            '$label: ',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(fontWeight: FontWeight.w600),
          ),
          Expanded(
            child: Text(
              filename,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.primary),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}

class _AdminPendingLawyersBanner extends StatelessWidget {
  const _AdminPendingLawyersBanner({required this.count, this.onReturn});

  final int count;
  final VoidCallback? onReturn;

  @override
  Widget build(BuildContext context) {
    const color = Color(0xFFE65100);
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: InkWell(
        onTap: () async {
          await Navigator.of(context).push(
            MaterialPageRoute<void>(builder: (_) => const AdminScreen(initialTab: 2)),
          );
          onReturn?.call();
        },
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.08),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withValues(alpha: 0.35)),
          ),
          child: Row(
            children: [
              const Icon(Icons.gavel_outlined, color: color, size: 22),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '$count pending lawyer application${count == 1 ? '' : 's'}',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            color: color, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Tap to review in the Admin panel',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: color, size: 18),
            ],
          ),
        ),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  const _ActionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: LegatoLinkedInTheme.navActiveGold, size: 24),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
            ],
          ),
        ),
      ),
    );
  }
}
