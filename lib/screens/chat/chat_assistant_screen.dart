import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/analysis_id_picker.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

enum _AssistantMode { general, negotiation }

class ChatAssistantScreen extends StatefulWidget {
  const ChatAssistantScreen({super.key});

  @override
  State<ChatAssistantScreen> createState() => _ChatAssistantScreenState();
}

class _ChatAssistantScreenState extends State<ChatAssistantScreen> {
  final _controller = TextEditingController();
  final List<_Msg> _history = [];
  _AssistantMode _mode = _AssistantMode.general;
  int? _selectedAnalysisId;
  bool _busy = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onModeChanged(_AssistantMode mode) {
    if (_mode == mode) return;
    setState(() {
      _mode = mode;
      _history.clear();
      if (mode == _AssistantMode.general) {
        _selectedAnalysisId = null;
      }
    });
  }

  List<Map<String, dynamic>> _historyPayload({bool excludeLast = false}) {
    final items = excludeLast && _history.isNotEmpty ? _history.sublist(0, _history.length - 1) : _history;
    return [
      for (final m in items)
        {'role': m.user ? 'user' : 'assistant', 'content': m.text},
    ];
  }

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _busy) return;
    if (_mode == _AssistantMode.negotiation && _selectedAnalysisId == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Select a contract from your history for negotiation mode.')),
      );
      return;
    }

    setState(() {
      _history.add(_Msg(true, text));
      _controller.clear();
      _busy = true;
    });

    try {
      final legato = context.read<AppServices>().legato;
      final Map<String, dynamic> res;
      if (_mode == _AssistantMode.negotiation) {
        res = await legato.negotiationChat(
          message: text,
          analysisId: _selectedAnalysisId,
          history: _historyPayload(excludeLast: true),
        );
      } else {
        res = await legato.chatAssistant(
          message: text,
          history: _historyPayload(excludeLast: true).isEmpty ? null : _historyPayload(excludeLast: true),
        );
      }
      final reply = res['content']?.toString() ?? '';
      if (!mounted) return;
      setState(() => _history.add(_Msg(false, reply)));
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _history.add(_Msg(false, 'Error: ${e.message}')));
    } catch (e) {
      if (!mounted) return;
      setState(() => _history.add(_Msg(false, 'Error: $e')));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final negotiation = _mode == _AssistantMode.negotiation;

    return Scaffold(
      appBar: LegatoAppBar(title: const Text('AI Assistant')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
            child: SegmentedButton<_AssistantMode>(
              segments: const [
                ButtonSegment(
                  value: _AssistantMode.general,
                  label: Text('General'),
                  icon: Icon(Icons.chat_outlined, size: 18),
                ),
                ButtonSegment(
                  value: _AssistantMode.negotiation,
                  label: Text('Negotiation'),
                  icon: Icon(Icons.handshake_outlined, size: 18),
                ),
              ],
              selected: {_mode},
              onSelectionChanged: (s) => _onModeChanged(s.first),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
            child: Text(
              negotiation
                  ? 'Strategy and talking points for your contract. Pick a saved analysis below.'
                  : 'General legal Q&A — contracts, labor law, and terminology.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                  ),
            ),
          ),
          if (negotiation)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
              child: AnalysisIdPicker(
                label: 'Contract from your history',
                onChanged: (id) => setState(() => _selectedAnalysisId = id),
              ),
            ),
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(12),
              itemCount: _history.length,
              itemBuilder: (context, i) {
                final m = _history[i];
                return Align(
                  alignment: m.user ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: const EdgeInsets.symmetric(vertical: 4),
                    padding: const EdgeInsets.all(12),
                    constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.85),
                    decoration: BoxDecoration(
                      color: m.user
                          ? Theme.of(context).colorScheme.primaryContainer
                          : Theme.of(context).colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: SelectableText(m.text),
                  ),
                );
              },
            ),
          ),
          if (_busy) const LinearProgressIndicator(minHeight: 2),
          Padding(
            padding: EdgeInsets.only(
              left: 8,
              right: 8,
              top: 8,
              bottom: MediaQuery.viewInsetsOf(context).bottom + 8,
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: negotiation ? 'Ask how to negotiate…' : 'Ask a legal question…',
                      border: const OutlineInputBorder(),
                    ),
                    minLines: 1,
                    maxLines: 4,
                    onSubmitted: (_) => _send(),
                  ),
                ),
                IconButton.filled(
                  onPressed: _busy ? null : _send,
                  icon: const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Msg {
  _Msg(this.user, this.text);
  final bool user;
  final String text;
}
