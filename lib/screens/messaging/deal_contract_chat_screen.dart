import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/mixins/message_poll_mixin.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/screens/messaging/deal_room_members_sheet.dart';
import 'package:legato_mobile/widgets/user_chat_bubble.dart';

/// Community discussion rooms (employment contract topics).
class DealContractChatScreen extends StatefulWidget {
  const DealContractChatScreen({super.key, this.embedded = false});

  final bool embedded;

  static const employmentCategory = 'employment';

  @override
  State<DealContractChatScreen> createState() => DealContractChatScreenState();
}

class DealContractChatScreenState extends State<DealContractChatScreen> with MessagePollMixin {
  List<dynamic> _threads = [];
  List<dynamic> _messages = [];
  int? _threadId;
  int? _threadCreatedBy;
  String? _activeRoomTitle;
  bool _loading = true;
  bool _sending = false;
  String? _err;
  final _body = TextEditingController();
  final _scroll = ScrollController();
  int _lastMessageCount = 0;

  @override
  void initState() {
    super.initState();
    _loadRooms();
  }

  Future<void> refresh() async {
    await _loadRooms();
    if (_threadId != null) {
      await _loadMessages(_threadId!, silent: true);
    }
  }

  @override
  void dispose() {
    _body.dispose();
    _scroll.dispose();
    super.dispose();
  }

  int? get _myId => context.read<AuthProvider>().user?.id;

  bool _isRoomMember(Map<String, dynamic> raw) {
    if (raw['is_member'] == true) return true;
    final createdBy = (raw['created_by'] as num?)?.toInt();
    if (_myId != null && createdBy == _myId) return true;
    return false;
  }

  bool get _canRenameRoom => _threadId != null && _threadCreatedBy != null && _threadCreatedBy == _myId;

  void _leaveRoom() {
    stopMessagePolling();
    setState(() {
      _threadId = null;
      _threadCreatedBy = null;
      _activeRoomTitle = null;
      _messages = [];
      _lastMessageCount = 0;
    });
    _loadRooms();
  }

  void _startThreadPolling() {
    if (_threadId == null) {
      stopMessagePolling();
      return;
    }
    startMessagePolling(() => _loadMessages(_threadId!, silent: true));
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

  Future<String?> _promptRoomName({String? initial, required String title}) async {
    final ctrl = TextEditingController(text: initial ?? '');
    final result = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(title),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          maxLength: 80,
          decoration: const InputDecoration(
            hintText: 'e.g. Doctors employment contracts',
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () {
              final t = ctrl.text.trim();
              if (t.length >= 3) Navigator.pop(ctx, t);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    ctrl.dispose();
    return result;
  }

  Future<void> _loadRooms() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final threads = await context.read<AppServices>().legato.listDealThreadsByCategory(
            DealContractChatScreen.employmentCategory,
          );
      if (!mounted) return;
      setState(() {
        _threads = threads;
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

  Future<void> _startThread() async {
    final name = await _promptRoomName(title: 'Name your room');
    if (name == null || !mounted) return;
    setState(() => _err = null);
    try {
      final t = await context.read<AppServices>().legato.createDealThreadByCategory(
            contractCategory: DealContractChatScreen.employmentCategory,
            title: name,
          );
      if (!mounted) return;
      final tid = (t['id'] as num?)?.toInt();
      await _loadRooms();
      if (tid != null) await _joinRoom(tid, name);
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _err = e.message);
    }
  }

  Future<void> _renameRoom() async {
    if (_threadId == null) return;
    final name = await _promptRoomName(
      title: 'Rename room',
      initial: _activeRoomTitle,
    );
    if (name == null || !mounted) return;
    try {
      await context.read<AppServices>().legato.updateDealThreadTitle(_threadId!, name);
      if (!mounted) return;
      setState(() => _activeRoomTitle = name);
      await _loadRooms();
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _err = e.message);
    }
  }

  Future<void> _joinRoom(int threadId, String title) async {
    setState(() {
      _activeRoomTitle = title;
      _threadId = threadId;
    });
    try {
      await context.read<AppServices>().legato.joinDealThread(threadId);
    } on ApiException catch (_) {
      // Room may still work if backend is not updated yet.
    }
    await _loadMessages(threadId);
  }

  void _showMembers() {
    if (_threadId == null) return;
    showDealRoomMembersSheet(
      context,
      threadId: _threadId!,
      roomTitle: _activeRoomTitle ?? 'Room',
      createdBy: _threadCreatedBy,
    );
  }

  Future<void> _loadMessages(int threadId, {bool silent = false}) async {
    try {
      final m = await context.read<AppServices>().legato.listDealMessages(threadId);
      if (!mounted) return;
      Map<String, dynamic>? threadMeta;
      for (final raw in _threads) {
        if (raw is Map && (raw['id'] as num?)?.toInt() == threadId) {
          threadMeta = Map<String, dynamic>.from(raw);
          break;
        }
      }
      final grew = m.length > _lastMessageCount;
      setState(() {
        _threadId = threadId;
        _threadCreatedBy = (threadMeta?['created_by'] as num?)?.toInt();
        _activeRoomTitle = threadMeta?['title']?.toString() ?? _activeRoomTitle ?? 'Room';
        _messages = m;
        _lastMessageCount = m.length;
        if (!silent) _err = null;
      });
      if (grew) _scrollToBottomIfNeeded(force: !silent);
      _startThreadPolling();
    } on ApiException catch (e) {
      if (!mounted) return;
      if (!silent) setState(() => _err = e.message);
    }
  }

  Future<void> _send() async {
    if (_threadId == null || _body.text.trim().isEmpty || _sending) return;
    setState(() => _sending = true);
    try {
      final text = _body.text.trim();
      _body.clear();
      await context.read<AppServices>().legato.postDealMessage(_threadId!, text);
      if (!mounted) return;
      await _loadMessages(_threadId!, silent: true);
      _scrollToBottomIfNeeded(force: true);
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _err = e.message);
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Widget _roomList() {
    if (_loading && _threads.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 4),
          child: Row(
            children: [
              TextButton(onPressed: _loading ? null : _startThread, child: const Text('New room')),
              const Spacer(),
              Text(
                '${_threads.length} room${_threads.length == 1 ? '' : 's'}',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                    ),
              ),
            ],
          ),
        ),
        if (_err != null)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ),
        Expanded(
          child: _threads.isEmpty
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Text(
                      'No community rooms yet. Tap New room to start a discussion.',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          ),
                    ),
                  ),
                )
              : RefreshIndicator(
                  onRefresh: refresh,
                  child: ListView.separated(
                    padding: const EdgeInsets.fromLTRB(12, 8, 12, 24),
                    itemCount: _threads.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 10),
                    itemBuilder: (context, i) {
                      final raw = Map<String, dynamic>.from(_threads[i] as Map);
                      final tid = (raw['id'] as num?)?.toInt();
                      final title = raw['title']?.toString() ?? 'Room #$tid';
                      if (tid == null) return const SizedBox.shrink();
                      final joined = _isRoomMember(raw);
                      return Card(
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                          side: BorderSide(
                            color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.35),
                          ),
                        ),
                        child: InkWell(
                          borderRadius: BorderRadius.circular(12),
                          onTap: joined ? () => _joinRoom(tid, title) : null,
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                            child: Row(
                              children: [
                                Icon(Icons.forum_outlined, color: LegatoLinkedInTheme.navActiveGold),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Text(
                                    title,
                                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                          fontWeight: FontWeight.w600,
                                        ),
                                  ),
                                ),
                                if (joined)
                                  TextButton(
                                    onPressed: () => _joinRoom(tid, title),
                                    child: const Text('Open'),
                                  )
                                else
                                  FilledButton(
                                    style: FilledButton.styleFrom(
                                      backgroundColor: LegatoLinkedInTheme.navActiveGold,
                                      foregroundColor: const Color(0xFF1B1F23),
                                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                                    ),
                                    onPressed: () => _joinRoom(tid, title),
                                    child: const Text('Join'),
                                  ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
        ),
      ],
    );
  }

  Widget _chatView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
          decoration: BoxDecoration(
            border: Border(
              bottom: BorderSide(color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.25)),
            ),
          ),
          child: Row(
            children: [
              IconButton(
                tooltip: 'Back to rooms',
                icon: const Icon(Icons.arrow_back),
                onPressed: _leaveRoom,
              ),
              Expanded(
                child: Text(
                  _activeRoomTitle ?? 'Room',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (_canRenameRoom)
                IconButton(
                  tooltip: 'Rename room',
                  icon: const Icon(Icons.edit_outlined),
                  onPressed: _renameRoom,
                ),
              IconButton(
                tooltip: 'Room members',
                icon: const Icon(Icons.group_outlined),
                onPressed: _showMembers,
              ),
            ],
          ),
        ),
        if (_err != null)
          Padding(
            padding: const EdgeInsets.all(8),
            child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ),
        Expanded(
          child: _messages.isEmpty
              ? Center(
                  child: Text(
                    'No messages yet. Say hello!',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                        ),
                  ),
                )
              : ListView.builder(
                  controller: _scroll,
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  itemCount: _messages.length,
                  itemBuilder: (context, i) {
                    final m = Map<String, dynamic>.from(_messages[i] as Map);
                    final email = m['email']?.toString() ?? '';
                    final authorId = (m['author_id'] as num?)?.toInt();
                    final me = _myId;
                    return UserChatBubble(
                      showAuthor: true,
                      message: UserChatMessage(
                        body: m['body']?.toString() ?? '',
                        isMine: me != null && authorId == me,
                        authorName: email.isNotEmpty ? email.split('@').first : '?',
                        createdAt: m['created_at']?.toString(),
                      ),
                    );
                  },
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
                    decoration: const InputDecoration(
                      hintText: 'Write a message…',
                      border: OutlineInputBorder(),
                    ),
                    onSubmitted: (_) => _send(),
                  ),
                ),
                IconButton.filled(onPressed: _sending ? null : _send, icon: const Icon(Icons.send)),
              ],
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_threadId != null) {
      return _chatView();
    }
    return _roomList();
  }
}
