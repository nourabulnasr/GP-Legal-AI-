import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class ChatDocumentScreen extends StatefulWidget {
  const ChatDocumentScreen({super.key});

  @override
  State<ChatDocumentScreen> createState() => _ChatDocumentScreenState();
}

class _ChatDocumentScreenState extends State<ChatDocumentScreen> {
  final _analysisId = TextEditingController();
  final _contextCtrl = TextEditingController();
  final _msg = TextEditingController();
  final _scroll = ScrollController();
  bool _busy = false;
  String? _err;
  final List<Map<String, String>> _thread = [];

  @override
  void dispose() {
    _analysisId.dispose();
    _contextCtrl.dispose();
    _msg.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final message = _msg.text.trim();
    if (message.isEmpty || _busy) return;
    int? aid;
    final raw = _analysisId.text.trim();
    if (raw.isNotEmpty) {
      aid = int.tryParse(raw);
      if (aid == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Analysis id must be a number')),
        );
        return;
      }
    }
    setState(() {
      _busy = true;
      _err = null;
    });
    final prior = _thread
        .map(
          (m) => <String, dynamic>{
            'role': m['role']!,
            'content': m['content']!,
          },
        )
        .toList();
    try {
      final res = await context.read<AppServices>().legato.chatWithDocument(
            analysisId: aid,
            documentContext: _contextCtrl.text.trim().isEmpty ? null : _contextCtrl.text.trim(),
            message: message,
            history: prior.isEmpty ? null : prior,
          );
      final reply = res['content']?.toString() ?? '';
      if (!mounted) return;
      setState(() {
        _thread.add({'role': 'user', 'content': message});
        _thread.add({'role': 'assistant', 'content': reply});
        _msg.clear();
      });
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) {
          _scroll.jumpTo(_scroll.position.maxScrollExtent);
        }
      });
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Document chat (LFM)')),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: ListView(
                controller: _scroll,
                padding: const EdgeInsets.all(16),
                children: [
                  TextField(
                    controller: _analysisId,
                    keyboardType: TextInputType.number,
                    decoration: const InputDecoration(
                      labelText: 'Optional analysis id (from history)',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _contextCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Optional pasted contract context (if no analysis id)',
                      border: OutlineInputBorder(),
                    ),
                    minLines: 2,
                    maxLines: 6,
                  ),
                  const SizedBox(height: 16),
                  if (_thread.isEmpty)
                    Text(
                      'Multi-turn chat: each reply stays in context for follow-up questions.',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ..._thread.map((m) {
                    final isUser = m['role'] == 'user';
                    return Align(
                      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                      child: Container(
                        margin: const EdgeInsets.symmetric(vertical: 6),
                        padding: const EdgeInsets.all(12),
                        constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.88),
                        decoration: BoxDecoration(
                          color: isUser
                              ? LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.22)
                              : Theme.of(context).colorScheme.surfaceContainerHighest,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: SelectableText(m['content'] ?? ''),
                      ),
                    );
                  }),
                  if (_err != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                    ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Expanded(
                    child: TextField(
                      controller: _msg,
                      decoration: const InputDecoration(
                        labelText: 'Message',
                        border: OutlineInputBorder(),
                      ),
                      minLines: 1,
                      maxLines: 4,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _send(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    style: FilledButton.styleFrom(
                      backgroundColor: LegatoLinkedInTheme.navActiveGold,
                      foregroundColor: const Color(0xFF1B1F23),
                    ),
                    onPressed: _busy ? null : _send,
                    child: _busy
                        ? const SizedBox(
                            height: 22,
                            width: 22,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF1B1F23)),
                          )
                        : const Icon(Icons.send),
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
