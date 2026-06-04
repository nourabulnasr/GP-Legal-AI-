import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/mixins/message_poll_mixin.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/messaging/group_members_sheet.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:legato_mobile/widgets/user_chat_bubble.dart';

class ConversationScreen extends StatefulWidget {
  const ConversationScreen({
    super.key,
    required this.conversationId,
    required this.title,
    this.isGroup = false,
    this.createdBy,
  });

  final int conversationId;
  final String title;
  final bool isGroup;
  final int? createdBy;

  @override
  State<ConversationScreen> createState() => _ConversationScreenState();
}

class _ConversationScreenState extends State<ConversationScreen> with MessagePollMixin {
  final _body = TextEditingController();
  final _scroll = ScrollController();
  List<dynamic> _messages = [];
  late String _title;
  bool _loading = true;
  bool _sending = false;
  String? _err;
  int _lastMessageCount = 0;

  bool get _canRenameGroup {
    if (!widget.isGroup) return false;
    final me = context.read<AuthProvider>().user?.id;
    return me != null && widget.createdBy == me;
  }

  @override
  void initState() {
    super.initState();
    _title = widget.title;
    _load(showSpinner: true).then((_) {
      startMessagePolling(() => _load(showSpinner: false));
    });
  }

  @override
  void dispose() {
    _body.dispose();
    _scroll.dispose();
    super.dispose();
  }

  bool get _nearBottom {
    if (!_scroll.hasClients) return true;
    return _scroll.position.pixels >= _scroll.position.maxScrollExtent - 80;
  }

  void _scrollToBottomIfNeeded({bool force = false}) {
    if (!force && !_nearBottom) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    });
  }

  ChatDeliveryStatus _statusFrom(String? raw) {
    if (raw == 'seen') return ChatDeliveryStatus.seen;
    if (raw == 'sent') return ChatDeliveryStatus.sent;
    return ChatDeliveryStatus.none;
  }

  Future<void> _load({bool showSpinner = false}) async {
    if (showSpinner) {
      setState(() {
        _loading = true;
        _err = null;
      });
    }
    try {
      final res = await context.read<AppServices>().legato.listConversationMessages(widget.conversationId);
      if (!mounted) return;
      final items = (res['items'] as List<dynamic>?) ?? [];
      final grew = items.length > _lastMessageCount;
      setState(() {
        _messages = items;
        _lastMessageCount = items.length;
        if (showSpinner) _loading = false;
      });
      if (grew) _scrollToBottomIfNeeded(force: showSpinner);
    } on ApiException catch (e) {
      if (!mounted) return;
      if (showSpinner) {
        setState(() {
          _err = e.message;
          _loading = false;
        });
      }
    }
  }

  Future<void> _renameGroup() async {
    final ctrl = TextEditingController(text: _title);
    final next = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Rename group'),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          maxLength: 256,
          decoration: const InputDecoration(hintText: 'Group name'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () {
              final t = ctrl.text.trim();
              if (t.length >= 1) Navigator.pop(ctx, t);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    ctrl.dispose();
    if (next == null || !mounted) return;
    try {
      await context.read<AppServices>().legato.updateGroupTitle(widget.conversationId, next);
      if (!mounted) return;
      setState(() => _title = next);
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  Future<void> _send() async {
    final text = _body.text.trim();
    if (text.isEmpty || _sending) return;
    setState(() => _sending = true);
    try {
      _body.clear();
      await context.read<AppServices>().legato.postConversationMessage(widget.conversationId, text);
      if (!mounted) return;
      await _load(showSpinner: false);
      _scrollToBottomIfNeeded(force: true);
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: LegatoAppBar(
        title: widget.isGroup
            ? InkWell(
                onTap: () => showGroupMembersSheet(
                  context,
                  conversationId: widget.conversationId,
                  groupTitle: _title,
                  createdBy: widget.createdBy,
                ),
                child: Text(
                  _title,
                  style: const TextStyle(decoration: TextDecoration.underline, decorationStyle: TextDecorationStyle.dotted),
                ),
              )
            : Text(_title),
        actions: [
          if (widget.isGroup)
            IconButton(
              tooltip: 'Group members',
              icon: const Icon(Icons.groups_outlined),
              onPressed: () => showGroupMembersSheet(
                context,
                conversationId: widget.conversationId,
                groupTitle: _title,
                createdBy: widget.createdBy,
              ),
            ),
          if (_canRenameGroup)
            IconButton(
              tooltip: 'Rename group',
              icon: const Icon(Icons.edit_outlined),
              onPressed: _renameGroup,
            ),
        ],
      ),
      body: Column(
        children: [
          if (_err != null)
            Padding(
              padding: const EdgeInsets.all(8),
              child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : RefreshIndicator(
                    onRefresh: () => _load(showSpinner: true),
                    child: _messages.isEmpty
                        ? ListView(
                            children: const [
                              SizedBox(height: 80),
                              Center(child: Text('No messages yet. Say hello!')),
                            ],
                          )
                        : ListView.builder(
                            controller: _scroll,
                            padding: const EdgeInsets.symmetric(vertical: 8),
                            itemCount: _messages.length,
                            itemBuilder: (context, i) {
                              final m = Map<String, dynamic>.from(_messages[i] as Map);
                              return UserChatBubble(
                                showAuthor: widget.isGroup,
                                message: UserChatMessage(
                                  body: m['body']?.toString() ?? '',
                                  isMine: m['is_mine'] == true,
                                  authorName: m['author_name']?.toString(),
                                  createdAt: m['created_at']?.toString(),
                                  status: _statusFrom(m['status']?.toString()),
                                ),
                              );
                            },
                          ),
                  ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(8),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _body,
                      minLines: 1,
                      maxLines: 4,
                      decoration: const InputDecoration(
                        hintText: 'Type a message…',
                        border: OutlineInputBorder(),
                      ),
                      onSubmitted: (_) => _send(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    onPressed: _sending ? null : _send,
                    icon: _sending
                        ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.send),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
