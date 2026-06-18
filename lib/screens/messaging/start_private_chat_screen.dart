import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/screens/messaging/conversation_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:legato_mobile/widgets/user_avatar.dart';

/// Pick a connection and start or open a private chat.
class StartPrivateChatScreen extends StatefulWidget {
  const StartPrivateChatScreen({super.key});

  @override
  State<StartPrivateChatScreen> createState() => _StartPrivateChatScreenState();
}

class _StartPrivateChatScreenState extends State<StartPrivateChatScreen> {
  bool _loading = true;
  bool _busy = false;
  String? _err;
  List<dynamic> _connections = [];

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
      final res = await context.read<AppServices>().legato.getNetworkConnections();
      if (!mounted) return;
      setState(() {
        _connections = (res['items'] as List<dynamic>?) ?? [];
        _loading = false;
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.message;
        _loading = false;
      });
    }
  }

  Future<void> _openChat(Map<String, dynamic> c) async {
    if (_busy) return;
    final userId = (c['user_id'] as num?)?.toInt();
    if (userId == null) return;
    setState(() => _busy = true);
    try {
      final conv = await context.read<AppServices>().legato.createDirectConversation(userId);
      if (!mounted) return;
      final id = (conv['id'] as num?)?.toInt();
      final title = c['display_name']?.toString() ??
          c['name']?.toString() ??
          c['email']?.toString().split('@').first ??
          'Chat';
      if (id == null) {
        Navigator.of(context).pop();
        return;
      }
      await Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(
          builder: (_) => ConversationScreen(
            conversationId: id,
            title: title,
            createdBy: (conv['created_by'] as num?)?.toInt(),
          ),
        ),
      );
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: LegatoAppBar(title: const Text('Private chat')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _err != null
              ? Center(child: Text(_err!))
              : _connections.isEmpty
                  ? const Center(child: Text('No connections yet. Connect with people from Network.'))
                  : ListView.separated(
                      itemCount: _connections.length,
                      separatorBuilder: (_, __) => const Divider(height: 1),
                      itemBuilder: (context, i) {
                        final c = Map<String, dynamic>.from(_connections[i] as Map);
        final name = c['display_name']?.toString() ??
            c['name']?.toString() ??
            c['email']?.toString().split('@').first ??
            'Member';
                        return ListTile(
                          leading: UserAvatar(
                            radius: 22,
                            imageUrl: c['avatar_url']?.toString(),
                            name: name,
                          ),
                          title: Row(children: [
                            Flexible(child: Text(name, overflow: TextOverflow.ellipsis)),
                            if (c['is_verified_lawyer'] == true) ...[
                              const SizedBox(width: 4),
                              const Tooltip(message: 'Verified Lawyer', child: Icon(Icons.verified, size: 14, color: Color(0xFF0A66C2))),
                            ],
                          ]),
                          subtitle: Text(c['email']?.toString() ?? ''),
                          trailing: Icon(Icons.chat_bubble_outline, color: LegatoLinkedInTheme.navActiveGold),
                          onTap: _busy ? null : () => _openChat(c),
                        );
                      },
                    ),
    );
  }
}
