import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/providers/theme_notifier.dart';
import 'package:legato_mobile/providers/user_profile_provider.dart';
import 'package:legato_mobile/screens/features/features_hub_screen.dart';
import 'package:legato_mobile/screens/more/more_screen.dart';
import 'package:legato_mobile/screens/social/profile_documents_screen.dart';
import 'package:legato_mobile/screens/social/profile_skills_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/utils/platform_file_bytes.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => ProfileScreenState();
}

class ProfileScreenState extends State<ProfileScreen> {
  void refresh() => _load(silent: _data != null);
  bool _loading = true;
  String? _err;
  Map<String, dynamic>? _data;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load({bool silent = false}) async {
    final uid = context.read<AuthProvider>().user?.id;
    if (uid == null) {
      setState(() {
        _loading = false;
        _err = 'Not signed in';
      });
      return;
    }
    if (!silent) {
      setState(() {
        _loading = true;
        _err = null;
      });
    }
    try {
      final email = context.read<AuthProvider>().user?.email ?? '';
      final legato = context.read<AppServices>().legato;
      final results = await Future.wait<dynamic>([
        legato.getSocialProfileResilient(uid, email),
        _loadConnectionCount(legato),
      ]);
      final d = Map<String, dynamic>.from(results[0] as Map);
      final stats = Map<String, dynamic>.from((d['stats'] as Map?) ?? {});
      stats['connections'] = results[1] as int;
      d['stats'] = stats;
      if (!mounted) return;
      context.read<UserProfileProvider>().applyFromProfile(
            d,
            fallbackName: email.contains('@') ? email.split('@').first : email,
          );
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

  // Mirrors NetworkScreen._loadConnections: same endpoint, same dedup-by-user_id,
  // so the profile stat matches the network page exactly.
  Future<int> _loadConnectionCount(dynamic api) async {
    try {
      final connRes = await api.getNetworkConnections();
      final raw = (connRes['items'] as List<dynamic>?) ??
          (connRes['connections'] as List<dynamic>?) ??
          <dynamic>[];
      final seen = <int>{};
      for (final item in raw) {
        final uid =
            (Map<String, dynamic>.from(item as Map)['user_id'] as num?)?.toInt() ?? 0;
        if (uid > 0) seen.add(uid);
      }
      return seen.length;
    } on ApiException catch (e) {
      if (e.statusCode != 404) rethrow;
      return 0;
    } catch (_) {
      return 0;
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
    final profile = context.watch<UserProfileProvider>();
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
          clipBehavior: Clip.none,
          slivers: [
            SliverToBoxAdapter(
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _ProfileCoverBanner(
                        onSettings: () => Navigator.of(context).push(
                          MaterialPageRoute<void>(builder: (_) => const MoreScreen()),
                        ),
                      ),
                      Container(
                        color: Theme.of(context).colorScheme.surface,
                        padding: const EdgeInsets.fromLTRB(16, 52, 16, 20),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              crossAxisAlignment: CrossAxisAlignment.center,
                              children: [
                                Flexible(
                                  child: Text(
                                    displayName,
                                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                          fontWeight: FontWeight.w800,
                                          letterSpacing: -0.3,
                                        ),
                                  ),
                                ),
                                if (auth.user?.isVerifiedLawyer == true) ...[
                                  const SizedBox(width: 8),
                                  const Tooltip(
                                    message: 'Verified Lawyer',
                                    child: Icon(Icons.verified, color: Color(0xFF0A66C2), size: 24),
                                  ),
                                ],
                              ],
                            ),
                            if (d['title']?.toString().isNotEmpty == true ||
                                d['company']?.toString().isNotEmpty == true) ...[
                              const SizedBox(height: 4),
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
                                  Expanded(
                                    child: Text(
                                      d['location'].toString(),
                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                                          ),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                            if (auth.user?.isLawyerAccount != true) ...[
                              const SizedBox(height: 14),
                              Row(
                                children: [
                                  Expanded(child: _StatPill(value: '${stats['connections'] ?? 0}', label: 'Connections')),
                                  const SizedBox(width: 8),
                                  Expanded(child: _StatPill(value: '${stats['endorsements'] ?? 0}', label: 'Endorsements')),
                                  const SizedBox(width: 8),
                                  Expanded(child: _StatPill(value: '${stats['profile_views'] ?? 0}', label: 'Views')),
                                ],
                              ),
                            ],
                            const SizedBox(height: 16),
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
                                if (auth.user?.isLawyerAccount != true) ...[
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
                                      child: Text(FeaturesHubScreen.allToolsLabel),
                                    ),
                                  ),
                                ],
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  Positioned(
                    left: 16,
                    top: _ProfileCoverBanner.height - _ProfileAvatarTile.size / 2,
                    child: _ProfileAvatarTile(
                      avatarUrl: profile.avatarUrl,
                      initial: initial,
                      onTap: _uploadAvatar,
                    ),
                  ),
                ],
              ),
            ),

            SliverToBoxAdapter(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
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
                  _Section(
                    title: 'Skills',
                    action: TextButton(
                      onPressed: () async {
                        await _addSkill(context);
                        await _load();
                      },
                      child: const Text('+ Add'),
                    ),
                    child: skills.isEmpty
                        ? Text(
                            'No skills added yet.',
                            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                  color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                                ),
                          )
                        : Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: skills.map<Widget>((s) {
                              final label = s.toString();
                              return InputChip(
                                label: Text(label, style: const TextStyle(fontSize: 12)),
                                backgroundColor: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.08),
                                side: BorderSide(color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.25)),
                                onPressed: () async {
                                  await _editSkill(context, label);
                                  await _load();
                                },
                                onDeleted: () async {
                                  await _deleteSkill(context, label);
                                  await _load();
                                },
                              );
                            }).toList(),
                          ),
                  ),

                  const SizedBox(height: 8),

                  // ── Experience ────────────────────────────────────────
                  _Section(
                    title: 'Experience',
                    action: TextButton(
                      onPressed: () async {
                        await _showExperienceDialog(context);
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
                                      PopupMenuButton<String>(
                                        onSelected: (v) async {
                                          if (v == 'edit') {
                                            await _showExperienceDialog(context, index: i, initial: m);
                                            await _load();
                                          } else if (v == 'delete') {
                                            await _deleteExperience(context, i);
                                            await _load();
                                          }
                                        },
                                        itemBuilder: (_) => const [
                                          PopupMenuItem(value: 'edit', child: Text('Edit')),
                                          PopupMenuItem(value: 'delete', child: Text('Delete')),
                                        ],
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
                        await _showEducationDialog(context);
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
                                          color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.1),
                                          borderRadius: BorderRadius.circular(6),
                                        ),
                                        child: Icon(
                                          Icons.school_outlined,
                                          size: 22,
                                          color: LegatoLinkedInTheme.navActiveGold,
                                        ),
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
                                      PopupMenuButton<String>(
                                        onSelected: (v) async {
                                          if (v == 'edit') {
                                            await _showEducationDialog(context, index: i, initial: m);
                                            await _load();
                                          } else if (v == 'delete') {
                                            await _deleteEducation(context, i);
                                            await _load();
                                          }
                                        },
                                        itemBuilder: (_) => const [
                                          PopupMenuItem(value: 'edit', child: Text('Edit')),
                                          PopupMenuItem(value: 'delete', child: Text('Delete')),
                                        ],
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
                              color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.1),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Icon(
                              Icons.folder_outlined,
                              size: 20,
                              color: LegatoLinkedInTheme.navActiveGold,
                            ),
                          ),
                          title: const Text('My Documents', style: TextStyle(fontWeight: FontWeight.w600)),
                          subtitle: const Text('Contracts and legal documents'),
                          trailing: Icon(Icons.chevron_right, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                          onTap: () => Navigator.of(context).push(
                            MaterialPageRoute<void>(builder: (_) => const ProfileDocumentsScreen()),
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
    final bytes = await readPlatformFileBytes(file);
    if (bytes == null || bytes.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not read the selected photo. Try another image.')),
        );
      }
      return;
    }
    final legato = context.read<AppServices>().legato;
    try {
      final resp = await legato.uploadProfileAvatar(bytes, pickedImageFilename(file));
      final url = resp['avatar_url']?.toString().trim();
      if (mounted && url != null && url.isNotEmpty) {
        context.read<UserProfileProvider>().setAvatarUrl(url);
        setState(() {
          _data = Map<String, dynamic>.from(_data ?? {})..['avatar_url'] = url;
        });
      }
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

  Future<void> _showExperienceDialog(
    BuildContext context, {
    int? index,
    Map<String, dynamic>? initial,
  }) async {
    final title = TextEditingController(text: initial?['title']?.toString() ?? '');
    final company = TextEditingController(text: initial?['company']?.toString() ?? '');
    final startDate = TextEditingController(text: initial?['startDate']?.toString() ?? '');
    final endRaw = initial?['endDate']?.toString() ?? '';
    final endDate = TextEditingController(text: endRaw == 'Present' ? '' : endRaw);
    final description = TextEditingController(text: initial?['description']?.toString() ?? '');
    final isEdit = index != null;
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(isEdit ? 'Edit experience' : 'Add experience'),
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
                final api = context.read<AppServices>().legato;
                if (isEdit) {
                  await api.updateProfileExperience(
                    index!,
                    title: title.text.trim(),
                    company: company.text.trim(),
                    startDate: startDate.text.trim(),
                    endDate: endDate.text.trim().isEmpty ? null : endDate.text.trim(),
                    description: description.text.trim(),
                  );
                } else {
                  await api.addProfileExperience(
                    title: title.text.trim(),
                    company: company.text.trim(),
                    startDate: startDate.text.trim(),
                    endDate: endDate.text.trim().isEmpty ? null : endDate.text.trim(),
                    description: description.text.trim(),
                  );
                }
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

  Future<void> _deleteExperience(BuildContext context, int index) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete experience?'),
        content: const Text('This entry will be removed from your profile.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true || !context.mounted) return;
    try {
      await context.read<AppServices>().legato.deleteProfileExperience(index);
    } on ApiException catch (e) {
      if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  Future<void> _showEducationDialog(
    BuildContext context, {
    int? index,
    Map<String, dynamic>? initial,
  }) async {
    final school = TextEditingController(text: initial?['school']?.toString() ?? '');
    final degree = TextEditingController(text: initial?['degree']?.toString() ?? '');
    final year = TextEditingController(text: initial?['year']?.toString() ?? '');
    final isEdit = index != null;
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(isEdit ? 'Edit education' : 'Add education'),
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
                final api = context.read<AppServices>().legato;
                if (isEdit) {
                  await api.updateProfileEducation(
                    index!,
                    school: school.text.trim(),
                    degree: degree.text.trim(),
                    year: year.text.trim(),
                  );
                } else {
                  await api.addProfileEducation(
                    school: school.text.trim(),
                    degree: degree.text.trim(),
                    year: year.text.trim(),
                  );
                }
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

  Future<void> _deleteEducation(BuildContext context, int index) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete education?'),
        content: const Text('This entry will be removed from your profile.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true || !context.mounted) return;
    try {
      await context.read<AppServices>().legato.deleteProfileEducation(index);
    } on ApiException catch (e) {
      if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  Future<void> _addSkill(BuildContext context) async {
    final ctrl = TextEditingController();
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add skill'),
        content: TextField(controller: ctrl, decoration: const InputDecoration(labelText: 'Skill name')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () async {
              final t = ctrl.text.trim();
              if (t.isEmpty) return;
              try {
                final d = _data;
                if (d == null) return;
                final cur = ((d['skills'] as List<dynamic>?) ?? []).map((e) => e.toString()).toList();
                if (!cur.contains(t)) cur.add(t);
                await context.read<AppServices>().legato.putProfileResilient({'skills': cur});
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
    ctrl.dispose();
  }

  Future<void> _editSkill(BuildContext context, String current) async {
    final ctrl = TextEditingController(text: current);
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Edit skill'),
        content: TextField(controller: ctrl, decoration: const InputDecoration(labelText: 'Skill name')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () async {
              final t = ctrl.text.trim();
              if (t.isEmpty) return;
              try {
                final d = _data;
                if (d == null) return;
                final cur = ((d['skills'] as List<dynamic>?) ?? []).map((e) => e.toString()).toList();
                final idx = cur.indexOf(current);
                if (idx >= 0) cur[idx] = t;
                await context.read<AppServices>().legato.putProfileResilient({'skills': cur});
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
    ctrl.dispose();
  }

  Future<void> _deleteSkill(BuildContext context, String skill) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remove skill?'),
        content: Text('Remove "$skill" from your profile?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Remove')),
        ],
      ),
    );
    if (ok != true || !context.mounted) return;
    try {
      final d = _data;
      if (d == null) return;
      final cur = ((d['skills'] as List<dynamic>?) ?? []).map((e) => e.toString()).where((s) => s != skill).toList();
      await context.read<AppServices>().legato.putProfileResilient({'skills': cur});
    } on ApiException catch (e) {
      if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }
}

// ── Shared widgets ─────────────────────────────────────────────────────────

class _ProfileCoverBanner extends StatelessWidget {
  const _ProfileCoverBanner({required this.onSettings});

  static const double height = 132;
  final VoidCallback onSettings;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height,
      child: Stack(
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
          Positioned(
            top: 8,
            right: 8,
            child: IconButton(
              icon: const Icon(Icons.settings_outlined, color: Colors.white70),
              onPressed: onSettings,
            ),
          ),
        ],
      ),
    );
  }
}

class _ProfileAvatarTile extends StatelessWidget {
  const _ProfileAvatarTile({
    required this.avatarUrl,
    required this.initial,
    required this.onTap,
  });

  static const double size = 96;

  final String? avatarUrl;
  final String initial;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final surface = Theme.of(context).colorScheme.surface;
    return GestureDetector(
      onTap: onTap,
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Container(
            width: size,
            height: size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: surface, width: 4),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.18),
                  blurRadius: 12,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: ClipOval(
              child: _avatarBody(context),
            ),
          ),
          Positioned(
            bottom: 2,
            right: 2,
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
    );
  }

  Widget _avatarBody(BuildContext context) {
    final url = avatarUrl;
    if (url == null || url.isEmpty) {
      return ColoredBox(
        color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.14),
        child: Center(
          child: Text(
            initial,
            style: const TextStyle(
              fontSize: 34,
              fontWeight: FontWeight.w800,
              color: Color(0xFF8B7318),
            ),
          ),
        ),
      );
    }
    return Image.network(
      url,
      key: ValueKey(url),
      width: size,
      height: size,
      fit: BoxFit.cover,
      errorBuilder: (context, error, stackTrace) => ColoredBox(
        color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.14),
        child: Center(
          child: Text(
            initial,
            style: const TextStyle(
              fontSize: 34,
              fontWeight: FontWeight.w800,
              color: Color(0xFF8B7318),
            ),
          ),
        ),
      ),
      loadingBuilder: (context, child, progress) {
        if (progress == null) return child;
        return ColoredBox(
          color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.14),
          child: const Center(child: CircularProgressIndicator(strokeWidth: 2)),
        );
      },
    );
  }
}

class _StatPill extends StatelessWidget {
  const _StatPill({required this.value, required this.label});

  final String value;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      decoration: BoxDecoration(
        color: Theme.of(context).scaffoldBackgroundColor,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.3)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 15, color: Color(0xFF8B7318)),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            textAlign: TextAlign.center,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
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
