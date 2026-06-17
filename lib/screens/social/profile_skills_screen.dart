import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/social/social_constants.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

class ProfileSkillsScreen extends StatefulWidget {
  const ProfileSkillsScreen({super.key});

  @override
  State<ProfileSkillsScreen> createState() => _ProfileSkillsScreenState();
}

class _ProfileSkillsScreenState extends State<ProfileSkillsScreen> {
  bool _loading = true;
  String? _err;
  List<dynamic> _items = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final uid = context.read<AuthProvider>().user?.id;
    if (uid == null) return;
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final r = await context.read<AppServices>().legato.getEndorsements(uid);
      if (!mounted) return;
      setState(() {
        _items = (r['items'] as List<dynamic>?) ?? [];
        _loading = false;
      });
    } on ApiException catch (e) {
      if (mounted) {
        setState(() {
          // Route missing or no data: show empty list instead of blocking the screen.
          if (e.statusCode == 404) {
            _items = [];
            _err = null;
          } else {
            _err = e.message;
          }
          _loading = false;
        });
      }
    }
  }

  Future<void> _addSkillToProfile(String t) async {
    final uid = context.read<AuthProvider>().user?.id;
    if (uid == null) return;
    try {
      final email = context.read<AuthProvider>().user?.email ?? '';
      final d = await context.read<AppServices>().legato.getSocialProfileResilient(uid, email);
      if (!mounted) return;
      final cur = (d['skills'] as List<dynamic>?) ?? [];
      final next = <String>{...cur.map((e) => e.toString()), t}.toList();
      await context.read<AppServices>().legato.putProfileResilient({'skills': next});
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Added skill: $t')));
      }
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: LegatoAppBar(title: const Text('Skills & Endorsements')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  Text(
                    'Endorsements you received',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_items.isEmpty)
                    Text('No endorsements yet.', style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)))
                  else
                    ..._items.map((raw) {
                      final m = raw as Map<String, dynamic>;
                      final endorserName = m['endorser_name']?.toString() ?? '';
                      final endorserIsVl = m['endorser_is_verified_lawyer'] == true;
                      return Card(
                        child: ListTile(
                          title: Text(m['skill']?.toString() ?? ''),
                          subtitle: Row(
                            children: [
                              Text('From '),
                              Flexible(child: Text(endorserName, overflow: TextOverflow.ellipsis)),
                              if (endorserIsVl) ...[
                                const SizedBox(width: 4),
                                const Tooltip(message: 'Verified Lawyer', child: Icon(Icons.verified, size: 13, color: Color(0xFF0A66C2))),
                              ],
                            ],
                          ),
                        ),
                      );
                    }),
                  const SizedBox(height: 24),
                  Text(
                    'Suggest a skill to add to your profile',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: kLegalTopicTags.map((t) {
                      return ActionChip(
                        label: Text(t),
                        onPressed: () => _addSkillToProfile(t),
                      );
                    }).toList(),
                  ),
                ],
              ),
            ),
    );
  }
}
