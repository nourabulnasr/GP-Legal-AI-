import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/mixins/message_poll_mixin.dart';
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
  void refresh() => _refreshActiveTab(silent: true);
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
      setState(() {
        _conversations = (res['items'] as List<dynamic>?) ?? [];
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

  void _refreshActiveTab({bool silent = false}) {
    if (_tabs.index == 0) {
      _load(showSpinner: !silent);
    } else {
      _dealRoomsKey.currentState?.refresh();
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

  Widget _conversationTile(Map<String, dynamic> c) {
    final kind = c['kind']?.toString() ?? 'direct';
    final title = c['title']?.toString() ?? 'Chat';
    final subtitle = c['last_message']?.toString() ?? (kind == 'group' ? 'Group chat' : 'Direct chat');
    String? avatarUrl;
    String avatarName = title;
    if (kind == 'direct') {
      final peers = (c['peers'] as List<dynamic>?) ?? [];
      if (peers.isNotEmpty && peers.first is Map) {
        final p = Map<String, dynamic>.from(peers.first as Map);
        avatarUrl = p['avatar_url']?.toString();
        avatarName = p['name']?.toString() ?? avatarName;
      }
    }

    return InkWell(
      onTap: () => _openConversation(c),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            if (kind == 'group')
              CircleAvatar(
                radius: 22,
                backgroundColor: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.2),
                child: Icon(Icons.groups, color: LegatoLinkedInTheme.navActiveGold, size: 22),
              )
            else
              UserAvatar(radius: 22, imageUrl: avatarUrl, name: avatarName),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (kind == 'group')
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
                    )
                  else
                    Text(title, style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
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
            if (kind == 'group')
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
    return Scaffold(
      appBar: LegatoAppBar(
        title: const Text('Messages'),
        bottom: TabBar(
          controller: _tabs,
          tabs: const [
            Tab(text: 'Chats'),
            Tab(text: 'Community'),
          ],
        ),
        actions: [
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
          IconButton(onPressed: _refreshActiveTab, icon: const Icon(Icons.refresh)),
        ],
      ),
      body: TabBarView(
        controller: _tabs,
        children: [
          _loading
              ? const Center(child: CircularProgressIndicator())
              : _err != null
                  ? Center(child: Text(_err!))
                  : RefreshIndicator(
                      onRefresh: () => _load(showSpinner: true),
                      child: _conversations.isEmpty
                          ? ListView(
                              children: [
                                const SizedBox(height: 48),
                                Icon(Icons.chat_outlined, size: 48, color: LegatoLinkedInTheme.navActiveGold),
                                const SizedBox(height: 12),
                                const Center(
                                  child: Padding(
                                    padding: EdgeInsets.symmetric(horizontal: 24),
                                    child: Text(
                                      'Start a private chat or group with your connections.',
                                      textAlign: TextAlign.center,
                                    ),
                                  ),
                                ),
                              ],
                            )
                          : ListView.separated(
                              itemCount: _conversations.length,
                              separatorBuilder: (_, __) => const Divider(height: 1),
                              itemBuilder: (context, i) {
                                final c = Map<String, dynamic>.from(_conversations[i] as Map);
                                return _conversationTile(c);
                              },
                            ),
                    ),
          DealContractChatScreen(key: _dealRoomsKey, embedded: true),
        ],
      ),
    );
  }
}
