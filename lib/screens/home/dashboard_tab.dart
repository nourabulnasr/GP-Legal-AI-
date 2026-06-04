import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/providers/user_profile_provider.dart';
import 'package:legato_mobile/screens/analyze/analyze_screen.dart';
import 'package:legato_mobile/screens/features/features_hub_screen.dart';
import 'package:legato_mobile/screens/translate/translate_contract_screen.dart';
import 'package:legato_mobile/screens/chat/chat_hub_screen.dart';
import 'package:legato_mobile/screens/history/history_screen.dart';
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
  }
  bool? _connected;
  Timer? _connectionTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _checkConnection());
    _connectionTimer = Timer.periodic(const Duration(seconds: 45), (_) => _checkConnection());
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
                                  Text(
                                    displayName,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 18,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  if (user != null) ...[
                                    const SizedBox(height: 4),
                                    Text(
                                      user.role.toUpperCase(),
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
                      child: Card(
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
