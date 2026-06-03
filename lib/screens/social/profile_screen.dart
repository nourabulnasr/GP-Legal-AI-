import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/providers/theme_notifier.dart';
import 'package:legato_mobile/screens/features/features_hub_screen.dart';
import 'package:legato_mobile/screens/more/more_screen.dart';
import 'package:legato_mobile/screens/social/profile_documents_screen.dart';
import 'package:legato_mobile/screens/social/profile_recommendations_screen.dart';
import 'package:legato_mobile/screens/social/profile_skills_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  bool _loading = true;
  String? _err;
  Map<String, dynamic>? _data;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final uid = context.read<AuthProvider>().user?.id;
    if (uid == null) {
      setState(() {
        _loading = false;
        _err = 'Not signed in';
      });
      return;
    }
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final email = context.read<AuthProvider>().user?.email ?? '';
      final d = await context.read<AppServices>().legato.getSocialProfileResilient(uid, email);
      if (!mounted) return;
      setState(() {
        _data = d;
        _loading = false;
      });
    } on ApiException catch (e) {
      if (mounted) {
        setState(() {
          _err = e.message;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _err = e.toString();
          _loading = false;
        });
      }
    }
  }

  Future<void> _editProfile() async {
    final d = _data;
    if (d == null) return;
    final name = TextEditingController(text: d['display_name']?.toString() ?? '');
    final title = TextEditingController(text: d['title']?.toString() ?? '');
    final company = TextEditingController(text: d['company']?.toString() ?? '');
    final location = TextEditingController(text: d['location']?.toString() ?? '');
    final bio = TextEditingController(text: d['bio']?.toString() ?? '');
    final skills = TextEditingController(
      text: ((d['skills'] as List<dynamic>?) ?? []).join(', '),
    );

    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Edit profile'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(controller: name, decoration: const InputDecoration(labelText: 'Display name')),
              TextField(controller: title, decoration: const InputDecoration(labelText: 'Title')),
              TextField(controller: company, decoration: const InputDecoration(labelText: 'Company')),
              TextField(controller: location, decoration: const InputDecoration(labelText: 'Location')),
              TextField(controller: bio, decoration: const InputDecoration(labelText: 'About'), maxLines: 3),
              TextField(
                controller: skills,
                decoration: const InputDecoration(labelText: 'Skills (comma-separated)'),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: LegatoLinkedInTheme.navActiveGold,
              foregroundColor: const Color(0xFF1B1F23),
            ),
            onPressed: () async {
              final list = skills.text
                  .split(',')
                  .map((s) => s.trim())
                  .where((s) => s.isNotEmpty)
                  .toList();
              try {
                final app = context.read<AppServices>();
                await app.legato.putProfileResilient({
                  'displayName': name.text.trim(),
                  'title': title.text.trim(),
                  'company': company.text.trim(),
                  'location': location.text.trim(),
                  'bio': bio.text.trim(),
                  'skills': list,
                });
                if (!mounted) return;
                if (ctx.mounted) Navigator.pop(ctx);
                await _load();
              } on ApiException catch (e) {
                if (ctx.mounted) ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(content: Text(e.message)));
              } catch (e) {
                if (ctx.mounted) ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(content: Text('$e')));
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    name.dispose();
    title.dispose();
    company.dispose();
    location.dispose();
    bio.dispose();
    skills.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final email = auth.user?.email ?? '';

    if (_loading) {
      return ColoredBox(
        color: Theme.of(context).scaffoldBackgroundColor,
        child: Center(child: CircularProgressIndicator()),
      );
    }

    if (_err != null || _data == null) {
      return ColoredBox(
        color: Theme.of(context).scaffoldBackgroundColor,
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(_err ?? 'Could not load profile'),
                TextButton(onPressed: _load, child: const Text('Retry')),
              ],
            ),
          ),
        ),
      );
    }

    final d = _data!;
    final stats = (d['stats'] as Map<String, dynamic>?) ?? {};
    final skills = (d['skills'] as List<dynamic>?) ?? [];
    final experience = (d['experience'] as List<dynamic>?) ?? [];
    final education = (d['education'] as List<dynamic>?) ?? [];
    final displayName = d['display_name']?.toString().isNotEmpty == true
        ? d['display_name'].toString()
        : email;
    final initial = displayName.isNotEmpty ? displayName[0].toUpperCase() : '?';

    return ColoredBox(
      color: Theme.of(context).scaffoldBackgroundColor,
      child: RefreshIndicator(
        onRefresh: _load,
        child: CustomScrollView(
          slivers: [
            // ── Cover + AppBar ────────────────────────────────────────────
            SliverAppBar(
              pinned: true,
              expandedHeight: 160,
              backgroundColor: const Color(0xFF1B1F23),
              actions: [
                IconButton(
                  icon: const Icon(Icons.settings_outlined, color: Colors.white70),
                  onPressed: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(builder: (_) => const MoreScreen()),
                  ),
                ),
              ],
              flexibleSpace: FlexibleSpaceBar(
                background: Stack(
                  fit: StackFit.expand,
                  children: [
                    Container(
                      decoration: const BoxDecoration(
                        gradient: LinearGradient(
                          colors: [Color(0xFF0A1628), Color(0xFF1B2A3E), Color(0xFF8B7318)],
                          stops: [0.0, 0.6, 1.0],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                      ),
                    ),
                    // Subtle geometric texture
                    Positioned(
                      right: -30,
                      top: -30,
                      child: Container(
                        width: 180,
                        height: 180,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.07),
                        ),
                      ),
                    ),
                    const Positioned(
                      left: -40,
                      bottom: -20,
                      child: SizedBox(
                        width: 130,
                        height: 130,
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Color(0x07C9A227),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            SliverToBoxAdapter(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // ── Hero card ─────────────────────────────────────────
                  Container(
                    color: Theme.of(context).colorScheme.surface,
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Avatar overlapping cover
                        Transform.translate(
                          offset: const Offset(0, -44),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.end,
                            children: [
                              GestureDetector(
                                onTap: _uploadAvatar,
                                child: Stack(
                                  children: [
                                    Container(
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        border: Border.all(color: Theme.of(context).colorScheme.surface, width: 4),
                                        boxShadow: [
                                          BoxShadow(
                                            color: Colors.black.withValues(alpha: 0.18),
                                            blurRadius: 12,
                                            offset: const Offset(0, 4),
                                          ),
                                        ],
                                      ),
                                      child: CircleAvatar(
                                        radius: 46,
                                        backgroundColor: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.14),
                                        backgroundImage: (d['avatar_url']?.toString().isNotEmpty == true)
                                            ? NetworkImage(d['avatar_url'].toString())
                                            : null,
                                        child: (d['avatar_url']?.toString().isNotEmpty != true)
                                            ? Text(
                                                initial,
                                                style: const TextStyle(
                                                  fontSize: 34,
                                                  fontWeight: FontWeight.w800,
                                                  color: Color(0xFF8B7318),
                                                ),
                                              )
                                            : null,
                                      ),
                                    ),
                                    Positioned(
                                      bottom: 4,
                                      right: 4,
                                      child: Container(
                                        padding: const EdgeInsets.all(4),
                                        decoration: const BoxDecoration(
                                          color: Color(0xFFC9A227),
                                          shape: BoxShape.circle,
                                        ),
                                        child: const Icon(Icons.camera_alt, size: 14, color: Color(0xFF1B1F23)),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const Spacer(),
                              Padding(
                                padding: const EdgeInsets.only(bottom: 4),
                                child: IconButton.filledTonal(
                                  icon: const Icon(Icons.edit_outlined, size: 20),
                                  onPressed: _editProfile,
                                  tooltip: 'Edit profile',
                                ),
                              ),
                            ],
                          ),
                        ),
                        // Name + headline
                        Transform.translate(
                          offset: const Offset(0, -28),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                displayName,
                                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                      fontWeight: FontWeight.w800,
                                      letterSpacing: -0.3,
                                    ),
                              ),
                              if (d['title']?.toString().isNotEmpty == true || d['company']?.toString().isNotEmpty == true) ...[
                                const SizedBox(height: 2),
                                Text(
                                  '${d['title'] ?? ''}${(d['title']?.toString().isNotEmpty == true) && (d['company']?.toString().isNotEmpty == true) ? ' · ' : ''}${d['company'] ?? ''}',
                                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                                        color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                                      ),
                                ),
                              ],
                              if (d['location']?.toString().isNotEmpty == true) ...[
                                const SizedBox(height: 4),
                                Row(
                                  children: [
                                    Icon(Icons.place_outlined, size: 15, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                                    const SizedBox(width: 3),
                                    Text(
                                      d['location'].toString(),
                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                                    ),
                                  ],
                                ),
                              ],
                              const SizedBox(height: 16),
                              // Stats row
                              Row(
                                children: [
                                  _StatPill(value: '${stats['connections'] ?? 0}', label: 'Connections'),
                                  const SizedBox(width: 8),
                                  _StatPill(value: '${stats['endorsements'] ?? 0}', label: 'Endorsements'),
                                  const SizedBox(width: 8),
                                  _StatPill(value: '${stats['profile_views'] ?? 0}', label: 'Views'),
                                ],
                              ),
                              const SizedBox(height: 16),
                              // Action buttons
                              Row(
                                children: [
                                  Expanded(
                                    child: FilledButton.icon(
                                      style: FilledButton.styleFrom(
                                        backgroundColor: LegatoLinkedInTheme.navActiveGold,
                                        foregroundColor: const Color(0xFF1B1F23),
                                        padding: const EdgeInsets.symmetric(vertical: 12),
                                      ),
                                      onPressed: _editProfile,
                                      icon: const Icon(Icons.edit_outlined, size: 16),
                                      label: const Text('Edit Profile'),
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: OutlinedButton(
                                      style: OutlinedButton.styleFrom(
                                        padding: const EdgeInsets.symmetric(vertical: 12),
                                        side: const BorderSide(color: Color(0xFF8B7318)),
                                      ),
                                      onPressed: () => Navigator.of(context).push(
                                        MaterialPageRoute<void>(builder: (_) => const FeaturesHubScreen()),
                                      ),
                                      child: const Text('All tools'),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 8),

                  // ── About ─────────────────────────────────────────────
                  _Section(
                    title: 'About',
                    action: TextButton(onPressed: _editProfile, child: const Text('Edit')),
                    child: Text(
                      d['bio']?.toString().isNotEmpty == true
                          ? d['bio'].toString()
                          : 'Add a short bio to let your network know who you are.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: d['bio']?.toString().isNotEmpty == true ? null : LegatoLinkedInTheme.textSecondaryAdaptive(context),
                            height: 1.55,
                          ),
                    ),
                  ),

                  const SizedBox(height: 8),

                  // ── Skills ────────────────────────────────────────────
                  if (skills.isNotEmpty)
                    _Section(
                      title: 'Skills',
                      action: TextButton(
                        onPressed: () => Navigator.of(context).push(
                          MaterialPageRoute<void>(builder: (_) => const ProfileSkillsScreen()),
                        ),
                        child: const Text('See all'),
                      ),
                      child: Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: skills.take(8).map<Widget>((s) {
                          return Chip(
                            label: Text(s.toString(), style: const TextStyle(fontSize: 12)),
                            backgroundColor: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.08),
                            side: BorderSide(color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.25)),
                          );
                        }).toList(),
                      ),
                    ),

                  if (skills.isNotEmpty) const SizedBox(height: 8),

                  // ── Experience ────────────────────────────────────────
                  _Section(
                    title: 'Experience',
                    action: TextButton(
                      onPressed: () async {
                        await _addExperience(context);
                        await _load();
                      },
                      child: const Text('+ Add'),
                    ),
                    child: experience.isEmpty
                        ? Text('No experience added yet.', style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)))
                        : Column(
                            children: experience.indexed.map((item) {
                              final (i, raw) = item;
                              final m = Map<String, dynamic>.from(raw as Map);
                              return Column(
                                children: [
                                  if (i > 0) const Divider(height: 24),
                                  Row(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Container(
                                        width: 42,
                                        height: 42,
                                        decoration: BoxDecoration(
                                          color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.1),
                                          borderRadius: BorderRadius.circular(6),
                                        ),
                                        child: const Icon(Icons.work_outline, size: 22, color: Color(0xFF8B7318)),
                                      ),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment: CrossAxisAlignment.start,
                                          children: [
                                            Text(m['title']?.toString() ?? '', style: Theme.of(context).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600)),
                                            if (m['company']?.toString().isNotEmpty == true)
                                              Text(m['company'].toString(), style: Theme.of(context).textTheme.bodySmall),
                                            Text(
                                              '${m['startDate'] ?? ''} – ${m['endDate'] ?? 'Present'}',
                                              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                                            ),
                                            if (m['description']?.toString().isNotEmpty == true) ...[
                                              const SizedBox(height: 4),
                                              Text(m['description'].toString(), style: Theme.of(context).textTheme.bodySmall),
                                            ],
                                          ],
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                              );
                            }).toList(),
                          ),
                  ),

                  const SizedBox(height: 8),

                  // ── Education ─────────────────────────────────────────
                  _Section(
                    title: 'Education',
                    action: TextButton(
                      onPressed: () async {
                        await _addEducation(context);
                        await _load();
                      },
                      child: const Text('+ Add'),
                    ),
                    child: education.isEmpty
                        ? Text('No education added yet.', style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)))
                        : Column(
                            children: education.indexed.map((item) {
                              final (i, raw) = item;
                              final m = Map<String, dynamic>.from(raw as Map);
                              return Column(
                                children: [
                                  if (i > 0) const Divider(height: 24),
                                  Row(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Container(
                                        width: 42,
                                        height: 42,
                                        decoration: BoxDecoration(
                                          color: const Color(0xFF1B2A3E).withValues(alpha: 0.08),
                                          borderRadius: BorderRadius.circular(6),
                                        ),
                                        child: const Icon(Icons.school_outlined, size: 22, color: Color(0xFF1B2A3E)),
                                      ),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment: CrossAxisAlignment.start,
                                          children: [
                                            Text(m['school']?.toString() ?? '', style: Theme.of(context).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600)),
                                            Text(
                                              '${m['degree'] ?? ''}${m['degree']?.toString().isNotEmpty == true && m['year']?.toString().isNotEmpty == true ? ' · ' : ''}${m['year'] ?? ''}',
                                              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                                            ),
                                          ],
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                              );
                            }).toList(),
                          ),
                  ),

                  const SizedBox(height: 8),

                  // ── More ──────────────────────────────────────────────
                  Container(
                    color: Theme.of(context).colorScheme.surface,
                    child: Column(
                      children: [
                        Consumer<ThemeNotifier>(
                          builder: (_, theme, _) => SwitchListTile(
                            secondary: Container(
                              width: 38,
                              height: 38,
                              decoration: BoxDecoration(
                                color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.1),
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: const Icon(Icons.dark_mode_outlined, size: 20, color: Color(0xFF8B7318)),
                            ),
                            title: const Text('Dark mode', style: TextStyle(fontWeight: FontWeight.w600)),
                            subtitle: const Text('Switch between light and dark'),
                            value: theme.isDark,
                            onChanged: (_) => theme.toggle(),
                          ),
                        ),
                        const Divider(height: 1, indent: 70),
                        ListTile(
                          leading: Container(
                            width: 38,
                            height: 38,
                            decoration: BoxDecoration(
                              color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.1),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: const Icon(Icons.workspace_premium_outlined, size: 20, color: Color(0xFF8B7318)),
                          ),
                          title: const Text('Skills & Endorsements', style: TextStyle(fontWeight: FontWeight.w600)),
                          subtitle: const Text('Manage skills your network can endorse'),
                          trailing: Icon(Icons.chevron_right, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                          onTap: () => Navigator.of(context).push(
                            MaterialPageRoute<void>(builder: (_) => const ProfileSkillsScreen()),
                          ),
                        ),
                        const Divider(height: 1, indent: 70),
                        ListTile(
                          leading: Container(
                            width: 38,
                            height: 38,
                            decoration: BoxDecoration(
                              color: const Color(0xFF1B2A3E).withValues(alpha: 0.08),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: const Icon(Icons.folder_outlined, size: 20, color: Color(0xFF1B2A3E)),
                          ),
                          title: const Text('My Documents', style: TextStyle(fontWeight: FontWeight.w600)),
                          subtitle: const Text('Contracts and legal documents'),
                          trailing: Icon(Icons.chevron_right, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                          onTap: () => Navigator.of(context).push(
                            MaterialPageRoute<void>(builder: (_) => const ProfileDocumentsScreen()),
                          ),
                        ),
                        const Divider(height: 1, indent: 70),
                        ListTile(
                          leading: Container(
                            width: 38,
                            height: 38,
                            decoration: BoxDecoration(
                              color: Colors.green.withValues(alpha: 0.08),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: const Icon(Icons.people_outline, size: 20, color: Colors.green),
                          ),
                          title: const Text('Recommendations', style: TextStyle(fontWeight: FontWeight.w600)),
                          subtitle: const Text('Give and receive recommendations'),
                          trailing: Icon(Icons.chevron_right, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                          onTap: () => Navigator.of(context).push(
                            MaterialPageRoute<void>(builder: (_) => const ProfileRecommendationsScreen()),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),
                  const Divider(),
                  ListTile(
                    leading: const Icon(Icons.logout, color: Colors.red),
                    title: const Text('Log out', style: TextStyle(color: Colors.red)),
                    onTap: () async {
                      final confirm = await showDialog<bool>(
                        context: context,
                        builder: (ctx) => AlertDialog(
                          title: const Text('Log out'),
                          content: const Text('Are you sure you want to log out?'),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.pop(ctx, false),
                              child: const Text('Cancel'),
                            ),
                            TextButton(
                              onPressed: () => Navigator.pop(ctx, true),
                              child: const Text('Log out', style: TextStyle(color: Colors.red)),
                            ),
                          ],
                        ),
                      );
                      if (confirm == true && context.mounted) {
                        await context.read<AuthProvider>().logout();
                      }
                    },
                  ),
                  const SizedBox(height: 16),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _uploadAvatar() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true,
      allowMultiple: false,
    );
    if (result == null || result.files.isEmpty) return;
    final file = result.files.first;
    if (file.bytes == null) return;
    try {
      await context.read<AppServices>().legato.uploadProfileAvatar(file.bytes!, file.name);
      if (!mounted) return;
      await _load();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Profile picture updated')),
        );
      }
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  Future<void> _addExperience(BuildContext context) async {
    final title = TextEditingController();
    final company = TextEditingController();
    final startDate = TextEditingController();
    final endDate = TextEditingController();
    final description = TextEditingController();
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add experience'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(controller: title, decoration: const InputDecoration(labelText: 'Job title')),
              TextField(controller: company, decoration: const InputDecoration(labelText: 'Company')),
              TextField(controller: startDate, decoration: const InputDecoration(labelText: 'Start date (e.g. 2022)')),
              TextField(controller: endDate, decoration: const InputDecoration(labelText: 'End date (leave blank = Present)')),
              TextField(controller: description, decoration: const InputDecoration(labelText: 'Description'), maxLines: 2),
            ],
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () async {
              try {
                await context.read<AppServices>().legato.addProfileExperience(
                      title: title.text.trim(),
                      company: company.text.trim(),
                      startDate: startDate.text.trim(),
                      endDate: endDate.text.trim().isEmpty ? null : endDate.text.trim(),
                      description: description.text.trim(),
                    );
                if (ctx.mounted) Navigator.pop(ctx);
              } on ApiException catch (e) {
                if (ctx.mounted) ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(content: Text(e.message)));
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    title.dispose();
    company.dispose();
    startDate.dispose();
    endDate.dispose();
    description.dispose();
  }

  Future<void> _addEducation(BuildContext context) async {
    final school = TextEditingController();
    final degree = TextEditingController();
    final year = TextEditingController();
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add education'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(controller: school, decoration: const InputDecoration(labelText: 'School')),
            TextField(controller: degree, decoration: const InputDecoration(labelText: 'Degree')),
            TextField(controller: year, decoration: const InputDecoration(labelText: 'Year')),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () async {
              try {
                await context.read<AppServices>().legato.addProfileEducation(
                      school: school.text.trim(),
                      degree: degree.text.trim(),
                      year: year.text.trim(),
                    );
                if (ctx.mounted) Navigator.pop(ctx);
              } on ApiException catch (e) {
                if (ctx.mounted) ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(content: Text(e.message)));
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    school.dispose();
    degree.dispose();
    year.dispose();
  }
}

// ── Shared widgets ─────────────────────────────────────────────────────────

class _StatPill extends StatelessWidget {
  const _StatPill({required this.value, required this.label});

  final String value;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Theme.of(context).scaffoldBackgroundColor,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 15, color: Color(0xFF8B7318)),
          ),
          Text(
            label,
            style: TextStyle(fontSize: 10, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
          ),
        ],
      ),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.child, this.action});

  final String title;
  final Widget child;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Theme.of(context).colorScheme.surface,
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
                ),
              ),
              ?action,
            ],
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }
}
