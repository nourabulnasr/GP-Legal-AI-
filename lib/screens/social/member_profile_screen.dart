import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/messaging/conversation_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:legato_mobile/widgets/user_avatar.dart';

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
      final legato = context.read<AppServices>().legato;
      var d = await legato.getSocialProfileResilient(widget.userId, email);
      final status = d['connection_status']?.toString();
      if (status == null || status.isEmpty || status == 'none') {
        try {
          final conns = await legato.getNetworkConnections();
          final items = (conns['items'] as List<dynamic>?) ?? [];
          final connected = items.any(
            (c) => c is Map && (c['user_id'] as num?)?.toInt() == widget.userId,
          );
          if (connected) {
            d = Map<String, dynamic>.from(d)..['connection_status'] = 'connected';
          }
        } catch (_) {}
      }
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

  Future<void> _message() async {
    try {
      final name = _data?['display_name']?.toString() ?? 'Chat';
      final conv = await context.read<AppServices>().legato.createDirectConversation(widget.userId);
      if (!mounted) return;
      final id = (conv['id'] as num?)?.toInt();
      if (id == null) return;
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => ConversationScreen(conversationId: id, title: name),
        ),
      );
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final me = auth.user?.id;

    return Scaffold(
      appBar: LegatoAppBar(title: const Text('Profile')),
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
                      _data?['connection_status']?.toString() == 'connected'
                          ? Row(
                              children: [
                                Expanded(
                                  child: FilledButton.icon(
                                    onPressed: null,
                                    style: FilledButton.styleFrom(
                                      backgroundColor: const Color(0xFF2E7D32),
                                      disabledBackgroundColor: const Color(0xFF2E7D32),
                                      disabledForegroundColor: Colors.white,
                                    ),
                                    icon: const Icon(Icons.check, size: 16, color: Colors.white),
                                    label: const Text('Connected'),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: FilledButton.icon(
                                    onPressed: _message,
                                    icon: const Icon(Icons.chat_bubble_outline, size: 16),
                                    label: const Text('Message'),
                                  ),
                                ),
                              ],
                            )
                          : _inviteSent || (_data?['connection_status']?.toString() == 'pending')
                              ? OutlinedButton.icon(
                                  onPressed: null,
                                  style: OutlinedButton.styleFrom(
                                    foregroundColor: Colors.grey,
                                    side: const BorderSide(color: Colors.grey),
                                  ),
                                  icon: const Icon(Icons.check, size: 16),
                                  label: const Text('Invitation Sent'),
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

  bool get _isVerifiedLawyer {
    if (data['is_verified_lawyer'] == true) return true;
    return data['user_type']?.toString().toLowerCase() == 'lawyer' &&
        data['lawyer_status']?.toString().toLowerCase() == 'approved';
  }

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
                UserAvatar(
                  radius: 26,
                  imageUrl: data['avatar_url']?.toString(),
                  name: data['display_name']?.toString() ?? 'Member',
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          Flexible(
                            child: Text(
                              data['display_name']?.toString() ?? 'Member',
                              style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
                            ),
                          ),
                          if (_isVerifiedLawyer) ...[
                            const SizedBox(width: 6),
                            const Tooltip(
                              message: 'Verified Lawyer',
                              child: Icon(Icons.verified, color: Color(0xFF0A66C2), size: 20),
                            ),
                          ],
                        ],
                      ),
                      const SizedBox(height: 2),
                      Text(
                        '${data['title'] ?? ''}${(data['title']?.toString().isNotEmpty == true) && (data['company']?.toString().isNotEmpty == true) ? ' � ' : ''}${data['company'] ?? ''}',
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

