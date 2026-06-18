import 'dart:async';

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
  int? _sessionRemainingSec;
  Timer? _sessionTimer;
  bool _sessionExpired = false;
  bool _sessionWaiting = false;
  String? _sessionStartsAtLabel;

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
      _loadSession();
    });
  }

  Future<void> _loadSession() async {
    try {
      final s = await context.read<AppServices>().legato.getConsultationSession(widget.conversationId);
      if (!mounted) return;
      if (s['active'] == true) {
        final rem = (s['remaining_seconds'] as num?)?.toInt() ?? 0;
        setState(() {
          _sessionRemainingSec = rem;
          _sessionExpired = rem <= 0;
          _sessionWaiting = false;
          _sessionStartsAtLabel = null;
        });
        _sessionTimer?.cancel();
        _sessionTimer = Timer.periodic(const Duration(seconds: 1), (_) {
          if (!mounted) return;
          setState(() {
            if (_sessionRemainingSec != null && _sessionRemainingSec! > 0) {
              _sessionRemainingSec = _sessionRemainingSec! - 1;
            } else {
              _sessionExpired = true;
              _sessionTimer?.cancel();
              _loadSession();
            }
          });
        });
      } else if (s['waiting'] == true || s['status']?.toString() == 'confirmed') {
        setState(() {
          _sessionWaiting = true;
          _sessionStartsAtLabel = s['starts_at_label']?.toString() ?? s['message']?.toString();
          _sessionRemainingSec = null;
          _sessionExpired = false;
        });
        _sessionTimer?.cancel();
        _sessionTimer = Timer.periodic(const Duration(seconds: 15), (_) => _loadSession());
      } else if (s['expired'] == true || s['status']?.toString() == 'expired') {
        setState(() {
          _sessionExpired = true;
          _sessionWaiting = false;
          _sessionRemainingSec = null;
          _sessionStartsAtLabel = null;
        });
        _sessionTimer?.cancel();
      } else {
        setState(() {
          _sessionWaiting = false;
          _sessionStartsAtLabel = null;
        });
      }
    } catch (_) {}
  }

  String _formatRemaining(int sec) {
    final m = sec ~/ 60;
    final s = sec % 60;
    return '${m}m ${s.toString().padLeft(2, '0')}s';
  }

  @override
  void dispose() {
    _sessionTimer?.cancel();
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

  Future<void> _payOffer(int messageId, Map<String, dynamic> offerData) async {
    final title = offerData['service_title']?.toString() ?? 'Legal Service';
    final price = offerData['price'];
    final currency = offerData['currency']?.toString() ?? 'USD';
    final priceStr = price != null ? '$currency ${(price as num).toStringAsFixed(2)}' : '';

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.payment, color: Color(0xFF0A66C2), size: 20),
            SizedBox(width: 8),
            Text('Confirm Payment'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Service: $title'),
            if (priceStr.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text('Amount: $priceStr', style: const TextStyle(fontWeight: FontWeight.w700)),
            ],
            const SizedBox(height: 12),
            const Text('By confirming, you acknowledge your payment intent for this legal service.'),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: const Color(0xFF0A66C2)),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Confirm Pay'),
          ),
        ],
      ),
    );

    if (confirmed != true || !mounted) return;
    try {
      await context.read<AppServices>().legato.acceptOffer(messageId);
      if (!mounted) return;
      await _load(showSpinner: false);
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final chatLocked = _sessionExpired || _sessionWaiting;
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
          if (_sessionWaiting)
            Container(
              width: double.infinity,
              color: Theme.of(context).colorScheme.secondaryContainer.withValues(alpha: 0.5),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  const Icon(Icons.schedule, size: 18),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _sessionStartsAtLabel != null
                          ? 'Consultation starts at $_sessionStartsAtLabel. Messaging opens then.'
                          : 'Consultation is booked. Messaging opens at the scheduled time.',
                    ),
                  ),
                ],
              ),
            ),
          if (_sessionRemainingSec != null && !_sessionExpired)
            Container(
              width: double.infinity,
              color: Theme.of(context).colorScheme.primaryContainer.withValues(alpha: 0.5),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  const Icon(Icons.timer_outlined, size: 18),
                  const SizedBox(width: 8),
                  Text('Consultation time remaining: ${_formatRemaining(_sessionRemainingSec!)}'),
                ],
              ),
            ),
          if (_sessionExpired)
            Container(
              width: double.infinity,
              color: Theme.of(context).colorScheme.errorContainer.withValues(alpha: 0.4),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  Icon(Icons.lock_clock, size: 18, color: Theme.of(context).colorScheme.error),
                  const SizedBox(width: 8),
                  const Expanded(child: Text('Consultation done. This chat has ended.')),
                ],
              ),
            ),
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
                              final offerRaw = m['offer_data'];
                              final msgId = (m['id'] as num?)?.toInt();
                              final isMine = m['is_mine'] == true;
                              final isOffer = m['msg_type']?.toString() == 'lawyer_offer';
                              return UserChatBubble(
                                showAuthor: widget.isGroup,
                                message: UserChatMessage(
                                  body: m['body']?.toString() ?? '',
                                  isMine: isMine,
                                  authorName: m['author_name']?.toString(),
                                  createdAt: m['created_at']?.toString(),
                                  status: _statusFrom(m['status']?.toString()),
                                  msgType: m['msg_type']?.toString() ?? 'text',
                                  offerData: offerRaw is Map
                                      ? Map<String, dynamic>.from(offerRaw)
                                      : null,
                                  offerStatus: m['offer_status']?.toString(),
                                  authorIsVerified: m['author_is_verified_lawyer'] == true,
                                ),
                                onOfferPay: (!isMine && isOffer && msgId != null)
                                    ? () => _payOffer(msgId, offerRaw is Map ? Map<String, dynamic>.from(offerRaw) : {})
                                    : null,
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
                      enabled: !chatLocked,
                      minLines: 1,
                      maxLines: 4,
                      decoration: InputDecoration(
                        hintText: _sessionWaiting
                            ? 'Opens at scheduled time'
                            : chatLocked
                                ? 'Session ended'
                                : 'Type a message…',
                        border: const OutlineInputBorder(),
                      ),
                      onSubmitted: chatLocked ? null : (_) => _send(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    onPressed: (_sending || chatLocked) ? null : _send,
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
