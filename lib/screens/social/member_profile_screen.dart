import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class MemberProfileScreen extends StatefulWidget {
  const MemberProfileScreen({super.key, required this.userId});

  final int userId;

  @override
  State<MemberProfileScreen> createState() => _MemberProfileScreenState();
}

class _MemberProfileScreenState extends State<MemberProfileScreen> {
  // Persists sent invite IDs for the entire app session so the button
  // doesn't reset to "Connect" when the user re-opens this screen.
  static final _sentIds = <int>{};

  bool _loading = true;
  String? _err;
  Map<String, dynamic>? _data;
  bool get _inviteSent => _sentIds.contains(widget.userId);

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final email = context.read<AuthProvider>().user?.email ?? '';
      final d = await context.read<AppServices>().legato.getSocialProfileResilient(widget.userId, email);
      if (!mounted) return;
      setState(() {
        _data = d;
        _loading = false;
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.message;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _connect() async {
    try {
      await context.read<AppServices>().legato.sendNetworkInvite(widget.userId);
      if (!mounted) return;
      _sentIds.add(widget.userId);
      setState(() {});
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Invitation sent')));
    } on ApiException catch (e) {
      if (!mounted) return;
      final msg = e.message.toLowerCase();
      if (msg.contains('already') || msg.contains('pending') || msg.contains('connected')) {
        _sentIds.add(widget.userId);
        setState(() {});
      }
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final me = auth.user?.id;

    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  if (_data == null)
                    Text('Not available', style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)))
                  else ...[
                    _HeaderCard(data: _data!),
                    const SizedBox(height: 12),
                    if (me != null && me != widget.userId)
                      _inviteSent || (_data?['connection_status']?.toString() == 'connected') || (_data?['connection_status']?.toString() == 'pending')
                          ? OutlinedButton.icon(
                              onPressed: null,
                              icon: const Icon(Icons.check, size: 16),
                              label: Text(_data?['connection_status']?.toString() == 'connected' ? 'Connected' : 'Pending'),
                            )
                          : FilledButton(
                              style: FilledButton.styleFrom(
                                backgroundColor: LegatoLinkedInTheme.navActiveGold,
                                foregroundColor: const Color(0xFF1B1F23),
                              ),
                              onPressed: _connect,
                              child: const Text('Connect'),
                            ),
                  ],
                ],
              ),
            ),
    );
  }
}

class _HeaderCard extends StatelessWidget {
  const _HeaderCard({required this.data});

  final Map<String, dynamic> data;

  @override
  Widget build(BuildContext context) {
    final skills = (data['skills'] as List<dynamic>?) ?? [];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  radius: 26,
                  backgroundColor: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.14),
                  child: Text(
                    (data['display_name']?.toString().isNotEmpty == true)
                        ? data['display_name'].toString()[0].toUpperCase()
                        : '?',
                    style: const TextStyle(fontWeight: FontWeight.w800, color: Color(0xFF8B7318)),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        data['display_name']?.toString() ?? 'Member',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        '${data['title'] ?? ''}${(data['title']?.toString().isNotEmpty == true) && (data['company']?.toString().isNotEmpty == true) ? ' · ' : ''}${data['company'] ?? ''}',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                      ),
                      if ((data['location']?.toString().isNotEmpty ?? false))
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Row(
                            children: [
                                              Icon(Icons.place_outlined, size: 16, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                              const SizedBox(width: 4),
                              Text(
                                data['location'].toString(),
                                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
            if ((data['bio']?.toString().isNotEmpty ?? false)) ...[
              const SizedBox(height: 12),
              Text(data['bio'].toString()),
            ],
            if (skills.isNotEmpty) ...[
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: skills.map((s) => Chip(label: Text(s.toString(), style: const TextStyle(fontSize: 12)))).toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

