import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/mixins/message_poll_mixin.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/messaging/conversation_screen.dart';
import 'package:legato_mobile/screens/messaging/create_group_screen.dart';
import 'package:legato_mobile/screens/messaging/deal_contract_chat_screen.dart';
import 'package:legato_mobile/screens/messaging/group_members_sheet.dart';
import 'package:legato_mobile/screens/messaging/start_private_chat_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:legato_mobile/widgets/user_avatar.dart';

class MessagesHubScreen extends StatefulWidget {
  const MessagesHubScreen({super.key});

  @override
  State<MessagesHubScreen> createState() => MessagesHubScreenState();
}

class MessagesHubScreenState extends State<MessagesHubScreen> with SingleTickerProviderStateMixin, MessagePollMixin {
  void refresh() => _refreshActiveTab(silent: false);
  late final TabController _tabs = TabController(length: 2, vsync: this);
  final _dealRoomsKey = GlobalKey<DealContractChatScreenState>();
  bool _loading = true;
  String? _err;
  List<dynamic> _conversations = [];

  @override
  Duration get pollInterval => const Duration(seconds: 3);

  @override
  void initState() {
    super.initState();
    _load(showSpinner: true).then((_) {
      startMessagePolling(() => _load(showSpinner: false));
    });
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  Future<void> _load({bool showSpinner = false}) async {
    if (showSpinner) {
      setState(() {
        _loading = true;
        _err = null;
      });
    }
    try {
      final res = await context.read<AppServices>().legato.listConversations();
      if (!mounted) return;
      final items = (res['items'] as List<dynamic>?) ?? [];
      setState(() {
        _conversations = items;
        if (showSpinner) _loading = false;
      });
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

  // Dispatches the AppBar Refresh button to whichever tab is active.
  // Lawyers never see tab 1, so _tabs.index stays 0 for them and this
  // always falls through to _load() — same effect as the old single-tab refresh.
  void _refreshActiveTab({bool silent = false}) {
    if (_tabs.index == 0) {
      _load(showSpinner: !silent);
    } else {
      _dealRoomsKey.currentState?.refresh();
    }
  }

  String? _consultationStatus(Map<String, dynamic> c) {
    final consult = c['consultation'];
    if (consult is! Map) return null;
    return consult['status']?.toString();
  }

  bool _isConsultationDone(Map<String, dynamic> c) => _consultationStatus(c) == 'expired';

  String _subtitleFor(Map<String, dynamic> c) {
    switch (_consultationStatus(c)) {
      case 'expired':
        return 'Consultation done';
      case 'confirmed':
        return 'Scheduled consultation';
      case 'active':
        return 'Consultation chat';
      default:
        return c['last_message']?.toString() ?? 'Consultation chat';
    }
  }

  void _openConversation(Map<String, dynamic> c) {
    final id = (c['id'] as num?)?.toInt();
    if (id == null) return;
    Navigator.of(context)
        .push(
          MaterialPageRoute<void>(
            builder: (_) => ConversationScreen(
              conversationId: id,
              title: c['title']?.toString() ?? 'Chat',
              isGroup: c['kind']?.toString() == 'group',
              createdBy: (c['created_by'] as num?)?.toInt(),
            ),
          ),
        )
        .then((_) => _load(showSpinner: true));
  }

  void _showGroupMembers(Map<String, dynamic> c) {
    final id = (c['id'] as num?)?.toInt();
    if (id == null) return;
    showGroupMembersSheet(
      context,
      conversationId: id,
      groupTitle: c['title']?.toString() ?? 'Group',
      createdBy: (c['created_by'] as num?)?.toInt(),
    );
  }

  Future<void> _deleteConversation(Map<String, dynamic> c) async {
    final id = (c['id'] as num?)?.toInt();
    if (id == null) return;
    final title = c['title']?.toString() ?? 'this chat';
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remove chat?'),
        content: Text('Remove the ended consultation with $title from your Messages list?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true || !mounted) return;
    try {
      await context.read<AppServices>().legato.deleteConversation(id);
      if (!mounted) return;
      await _load(showSpinner: true);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Chat removed')));
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  Widget _conversationTile(Map<String, dynamic> c) {
    if ((c['kind']?.toString() ?? 'direct') == 'group') return _groupTile(c);
    final title = c['title']?.toString() ?? 'Chat';
    final done = _isConsultationDone(c);
    final subtitle = _subtitleFor(c);
    String? avatarUrl;
    String avatarName = title;
    final peers = (c['peers'] as List<dynamic>?) ?? [];
    if (peers.isNotEmpty && peers.first is Map) {
      final p = Map<String, dynamic>.from(peers.first as Map);
      avatarUrl = p['avatar_url']?.toString();
      avatarName = p['name']?.toString() ?? avatarName;
    }
    final peerIsVl = peers.isNotEmpty && peers.first is Map && (peers.first as Map)['is_verified_lawyer'] == true;

    return InkWell(
      onTap: () => _openConversation(c),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            UserAvatar(radius: 22, imageUrl: avatarUrl, name: avatarName),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Flexible(
                        child: Text(
                          title,
                          overflow: TextOverflow.ellipsis,
                          style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                        ),
                      ),
                      if (peerIsVl) ...[
                        const SizedBox(width: 4),
                        const Tooltip(
                          message: 'Verified Lawyer',
                          child: Icon(Icons.verified, size: 14, color: Color(0xFF0A66C2)),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: done
                              ? LegatoLinkedInTheme.textSecondaryAdaptive(context)
                              : LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          fontStyle: done ? FontStyle.italic : FontStyle.normal,
                        ),
                  ),
                ],
              ),
            ),
            if (done)
              IconButton(
                tooltip: 'Remove chat',
                icon: Icon(Icons.delete_outline, color: Theme.of(context).colorScheme.error),
                onPressed: () => _deleteConversation(c),
              ),
          ],
        ),
      ),
    );
  }

  Widget _groupTile(Map<String, dynamic> c) {
    final title = c['title']?.toString() ?? 'Chat';
    final subtitle = c['last_message']?.toString() ?? 'Group chat';
    return InkWell(
      onTap: () => _openConversation(c),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            CircleAvatar(
              radius: 22,
              backgroundColor: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.2),
              child: Icon(Icons.groups, color: LegatoLinkedInTheme.navActiveGold, size: 22),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  InkWell(
                    onTap: () => _showGroupMembers(c),
                    child: Text(
                      title,
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w600,
                            decoration: TextDecoration.underline,
                            decorationColor: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.6),
                          ),
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                        ),
                  ),
                ],
              ),
            ),
            IconButton(
              tooltip: 'View members',
              icon: const Icon(Icons.info_outline),
              onPressed: () => _showGroupMembers(c),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isLawyer = context.watch<AuthProvider>().user?.isLawyerAccount == true;
    return Scaffold(
      appBar: LegatoAppBar(
        title: const Text('Messages'),
        bottom: isLawyer
            ? null
            : TabBar(
                controller: _tabs,
                tabs: const [
                  Tab(text: 'Chats'),
                  Tab(text: 'Community'),
                ],
              ),
        actions: [
          if (!isLawyer) ...[
            IconButton(
              tooltip: 'Private chat',
              icon: const Icon(Icons.chat_outlined),
              onPressed: () async {
                await Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const StartPrivateChatScreen()),
                );
                if (mounted) _load(showSpinner: true);
              },
            ),
            IconButton(
              tooltip: 'New group',
              icon: const Icon(Icons.group_add_outlined),
              onPressed: () async {
                await Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const CreateGroupScreen()),
                );
                if (mounted) _load(showSpinner: true);
              },
            ),
          ],
          IconButton(onPressed: _refreshActiveTab, icon: const Icon(Icons.refresh)),
        ],
      ),
      body: isLawyer
          ? _buildChatsTab(isLawyer: true)
          : TabBarView(
              controller: _tabs,
              children: [
                _buildChatsTab(isLawyer: false),
                DealContractChatScreen(key: _dealRoomsKey, embedded: true),
              ],
            ),
    );
  }

  Widget _buildChatsTab({required bool isLawyer}) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_err != null) {
      return Center(child: Text(_err!));
    }
    // Lawyers see only direct (consultation) chats — groups are hidden at render
    // so we don't have to re-fetch when account type changes.
    final visible = isLawyer
        ? _conversations.where((raw) {
            if (raw is! Map) return false;
            return (raw['kind']?.toString() ?? 'direct') == 'direct';
          }).toList()
        : _conversations;
    return RefreshIndicator(
      onRefresh: () => _load(showSpinner: true),
      child: visible.isEmpty
          ? ListView(
              children: [
                const SizedBox(height: 48),
                Icon(Icons.chat_outlined, size: 48, color: LegatoLinkedInTheme.navActiveGold),
                const SizedBox(height: 12),
                Center(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 24),
                    child: Text(
                      isLawyer
                          ? 'Consultation chats appear here after a booking is confirmed.'
                          : 'Start a private chat or group with your connections.',
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
              ],
            )
          : ListView.separated(
              itemCount: visible.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (context, i) {
                final c = Map<String, dynamic>.from(visible[i] as Map);
                return _conversationTile(c);
              },
            ),
    );
  }
}
