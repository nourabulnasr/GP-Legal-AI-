import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/screens/messaging/conversation_screen.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

class CreateGroupScreen extends StatefulWidget {
  const CreateGroupScreen({super.key});

  @override
  State<CreateGroupScreen> createState() => _CreateGroupScreenState();
}

class _CreateGroupScreenState extends State<CreateGroupScreen> {
  final _title = TextEditingController(text: 'Group chat');
  bool _loading = true;
  bool _busy = false;
  String? _err;
  List<dynamic> _connections = [];
  final Set<int> _selected = {};

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _title.dispose();
    super.dispose();
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

  Future<void> _create() async {
    if (_selected.isEmpty) return;
    setState(() => _busy = true);
    try {
      final conv = await context.read<AppServices>().legato.createGroupConversation(
            title: _title.text.trim().isEmpty ? 'Group chat' : _title.text.trim(),
            memberIds: _selected.toList(),
          );
      if (!mounted) return;
      final id = (conv['id'] as num?)?.toInt();
      if (id != null) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute<void>(
            builder: (_) => ConversationScreen(
              conversationId: id,
              title: conv['title']?.toString() ?? 'Group',
              isGroup: true,
              createdBy: (conv['created_by'] as num?)?.toInt(),
            ),
          ),
        );
      } else {
        Navigator.of(context).pop();
      }
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
      appBar: LegatoAppBar(title: const Text('New group chat')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: TextField(
                    controller: _title,
                    decoration: const InputDecoration(
                      labelText: 'Group name',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                if (_err != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  ),
                Expanded(
                  child: _connections.isEmpty
                      ? const Center(child: Text('Connect with people first (Network tab).'))
                      : ListView.builder(
                          itemCount: _connections.length,
                          itemBuilder: (context, i) {
                            final m = Map<String, dynamic>.from(_connections[i] as Map);
                            final uid = (m['user_id'] as num?)?.toInt() ?? 0;
                            final name = m['name']?.toString() ?? 'Member';
                            return CheckboxListTile(
                              value: _selected.contains(uid),
                              onChanged: uid <= 0
                                  ? null
                                  : (v) => setState(() {
                                        if (v == true) {
                                          _selected.add(uid);
                                        } else {
                                          _selected.remove(uid);
                                        }
                                      }),
                              title: Row(
                                children: [
                                  Flexible(child: Text(name, overflow: TextOverflow.ellipsis)),
                                  if (m['is_verified_lawyer'] == true) ...[
                                    const SizedBox(width: 4),
                                    const Tooltip(message: 'Verified Lawyer', child: Icon(Icons.verified, size: 14, color: Color(0xFF0A66C2))),
                                  ],
                                ],
                              ),
                              subtitle: Text(m['subtitle']?.toString() ?? ''),
                            );
                          },
                        ),
                ),
                SafeArea(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: FilledButton(
                      onPressed: _busy || _selected.isEmpty ? null : _create,
                      child: Text(_busy ? 'Creating…' : 'Create group (${_selected.length})'),
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}
