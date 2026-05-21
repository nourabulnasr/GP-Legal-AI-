import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

/// POST /chat/document (LFM) — pre-selects an analysis_id from saved analyses.
class ChatAnalysisScreen extends StatefulWidget {
  const ChatAnalysisScreen({super.key, this.initialAnalysisId});

  final int? initialAnalysisId;

  @override
  State<ChatAnalysisScreen> createState() => _ChatAnalysisScreenState();
}

class _ChatAnalysisScreenState extends State<ChatAnalysisScreen> {
  final _msg = TextEditingController();
  final _scroll = ScrollController();
  int? _selectedId;
  List<_AnalysisItem> _analyses = [];
  bool _loadingAnalyses = false;
  bool _busy = false;
  String? _loadErr;
  String? _err;

  /// Conversation thread for multi-turn context.
  final List<Map<String, String>> _thread = [];

  @override
  void initState() {
    super.initState();
    _selectedId = widget.initialAnalysisId;
    _loadAnalyses();
  }

  Future<void> _loadAnalyses() async {
    setState(() => _loadingAnalyses = true);
    try {
      final list = await context.read<AppServices>().legato.listAnalyses();
      if (!mounted) return;
      setState(() {
        _analyses = list
            .whereType<Map>()
            .map((m) {
              final id = (m['id'] as num?)?.toInt();
              final name = m['filename']?.toString() ??
                  m['original_filename']?.toString() ??
                  'Analysis #$id';
              return id != null ? _AnalysisItem(id: id, label: name) : null;
            })
            .whereType<_AnalysisItem>()
            .toList();
        if (_selectedId != null && !_analyses.any((a) => a.id == _selectedId)) {
          _analyses.insert(0, _AnalysisItem(id: _selectedId!, label: 'Analysis #$_selectedId'));
        }
        if (_selectedId == null && _analyses.isNotEmpty) {
          _selectedId = _analyses.first.id;
        }
      });
    } on ApiException catch (e) {
      if (mounted) setState(() => _loadErr = e.message);
    } catch (e) {
      if (mounted) setState(() => _loadErr = e.toString());
    } finally {
      if (mounted) setState(() => _loadingAnalyses = false);
    }
  }

  @override
  void dispose() {
    _msg.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    if (_selectedId == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Select an analysis first')),
      );
      return;
    }
    final message = _msg.text.trim();
    if (message.isEmpty || _busy) return;

    final userMsg = message;
    _msg.clear();
    setState(() {
      _busy = true;
      _err = null;
      _thread.add({'role': 'user', 'content': userMsg});
    });

    final prior = _thread
        .sublist(0, _thread.length - 1)
        .map((m) => <String, dynamic>{'role': m['role']!, 'content': m['content']!})
        .toList();

    try {
      final res = await context.read<AppServices>().legato.chatWithDocument(
            analysisId: _selectedId!,
            message: userMsg,
            history: prior.isEmpty ? null : prior,
          );
      final reply = res['content']?.toString() ?? '';
      if (!mounted) return;
      setState(() => _thread.add({'role': 'assistant', 'content': reply}));
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _thread.removeLast();
        _err = e.message;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _thread.removeLast();
        _err = e.toString();
      });
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Document chat — LFM')),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
              child: _loadingAnalyses
                  ? const LinearProgressIndicator()
                  : _loadErr != null
                      ? Text(_loadErr!, style: TextStyle(color: Theme.of(context).colorScheme.error, fontSize: 13))
                      : _analyses.isEmpty
                          ? const Text('No saved analyses. Upload and analyze a contract first.')
                          : DropdownButtonFormField<int>(
                              key: ValueKey(_selectedId),
                              initialValue: _selectedId,
                              decoration: const InputDecoration(
                                labelText: 'Select analysis',
                                border: OutlineInputBorder(),
                              ),
                              items: _analyses
                                  .map((a) => DropdownMenuItem(
                                        value: a.id,
                                        child: Text(a.label, overflow: TextOverflow.ellipsis),
                                      ))
                                  .toList(),
                              onChanged: (v) => setState(() => _selectedId = v),
                            ),
            ),
            Expanded(
              child: ListView(
                controller: _scroll,
                padding: const EdgeInsets.all(16),
                children: [
                  if (_thread.isEmpty)
                    Text(
                      'Ask anything about the selected contract analysis. Runs on local LFM.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
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
                        labelText: 'Ask about the contract…',
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

class _AnalysisItem {
  const _AnalysisItem({required this.id, required this.label});
  final int id;
  final String label;
}
