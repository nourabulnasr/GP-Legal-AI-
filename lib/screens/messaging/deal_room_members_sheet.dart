import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/user_avatar.dart';

Future<void> showDealRoomMembersSheet(
  BuildContext context, {
  required int threadId,
  required String roomTitle,
  required int? createdBy,
}) {
  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    builder: (ctx) => _DealRoomMembersSheet(
      threadId: threadId,
      roomTitle: roomTitle,
      createdBy: createdBy,
    ),
  );
}

class _DealRoomMembersSheet extends StatefulWidget {
  const _DealRoomMembersSheet({
    required this.threadId,
    required this.roomTitle,
    required this.createdBy,
  });

  final int threadId;
  final String roomTitle;
  final int? createdBy;

  @override
  State<_DealRoomMembersSheet> createState() => _DealRoomMembersSheetState();
}

class _DealRoomMembersSheetState extends State<_DealRoomMembersSheet> {
  bool _loading = true;
  String? _err;
  List<Map<String, dynamic>> _members = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  String _nameFromEmail(String email) {
    if (email.contains('@')) return email.split('@').first;
    return email.trim();
  }

  Future<Map<String, dynamic>> _memberFromProfile(
    int userId, {
    required bool isCreator,
    String? emailHint,
  }) async {
    final legato = context.read<AppServices>().legato;
    final hint = emailHint ?? '';
    try {
      final d = await legato.getSocialProfileResilient(userId, hint);
      if (!mounted) {
        return {
          'user_id': userId,
          'name': 'Member',
          'email': hint,
          'avatar_url': '',
          'is_creator': isCreator,
        };
      }
      final name = (d['display_name']?.toString() ?? '').trim();
      final email = d['email']?.toString() ?? hint;
      final resolvedName = name.isNotEmpty
          ? name
          : (_nameFromEmail(email).isNotEmpty ? _nameFromEmail(email) : 'Member');
      return {
        'user_id': userId,
        'name': resolvedName,
        'email': email,
        'avatar_url': d['avatar_url']?.toString() ?? '',
        'is_creator': isCreator,
      };
    } catch (_) {
      final fallback = _nameFromEmail(hint);
      return {
        'user_id': userId,
        'name': fallback.isNotEmpty ? fallback : 'Member',
        'email': hint,
        'avatar_url': '',
        'is_creator': isCreator,
      };
    }
  }

  Future<List<Map<String, dynamic>>> _membersFromMessages(List<dynamic> msgs) async {
    final emailById = <int, String>{};
    for (final raw in msgs) {
      if (raw is! Map) continue;
      final m = Map<String, dynamic>.from(raw);
      final uid = (m['author_id'] as num?)?.toInt();
      if (uid == null) continue;
      final email = m['email']?.toString() ?? '';
      if (email.isNotEmpty || !emailById.containsKey(uid)) {
        emailById[uid] = email;
      }
    }

    final creatorId = widget.createdBy;
    if (creatorId != null) {
      emailById.putIfAbsent(creatorId, () => '');
    }

    final members = <Map<String, dynamic>>[];
    for (final entry in emailById.entries) {
      members.add(
        await _memberFromProfile(
          entry.key,
          isCreator: entry.key == creatorId,
          emailHint: entry.value,
        ),
      );
    }
    return _sortMembers(members);
  }

  Future<void> _enrichMembers(List<Map<String, dynamic>> items) async {
    for (var i = 0; i < items.length; i++) {
      final m = items[i];
      final uid = (m['user_id'] as num?)?.toInt();
      if (uid == null) continue;
      final name = m['name']?.toString() ?? '';
      final needsProfile = name.isEmpty ||
          name == 'Room creator' ||
          name == 'Member' ||
          (m['avatar_url']?.toString().isEmpty ?? true);
      if (!needsProfile) continue;
      items[i] = await _memberFromProfile(
        uid,
        isCreator: m['is_creator'] == true || uid == widget.createdBy,
        emailHint: m['email']?.toString(),
      );
    }
  }

  List<Map<String, dynamic>> _sortMembers(List<Map<String, dynamic>> items) {
    final sorted = List<Map<String, dynamic>>.from(items);
    sorted.sort((a, b) {
      final aCreator = a['is_creator'] == true;
      final bCreator = b['is_creator'] == true;
      if (aCreator != bCreator) return aCreator ? -1 : 1;
      return (a['name']?.toString() ?? '').compareTo(b['name']?.toString() ?? '');
    });
    return sorted;
  }

  Future<void> _loadFromMessages() async {
    final msgs = await context.read<AppServices>().legato.listDealMessages(widget.threadId);
    if (!mounted) return;
    final members = await _membersFromMessages(msgs);
    if (!mounted) return;
    setState(() {
      _members = members;
      _loading = false;
      _err = null;
    });
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      await context.read<AppServices>().legato.joinDealThread(widget.threadId);
    } catch (_) {}
    try {
      final res = await context.read<AppServices>().legato.listDealThreadMembers(widget.threadId);
      if (!mounted) return;
      final items = (res['items'] as List<dynamic>?) ?? [];
      final members = items.map((e) => Map<String, dynamic>.from(e as Map)).toList();
      final createdBy = (res['created_by'] as num?)?.toInt() ?? widget.createdBy;
      for (final m in members) {
        final uid = (m['user_id'] as num?)?.toInt();
        if (uid != null && uid == createdBy) {
          m['is_creator'] = true;
        }
      }
      if (createdBy != null &&
          !members.any((m) => (m['user_id'] as num?)?.toInt() == createdBy)) {
        members.insert(
          0,
          await _memberFromProfile(createdBy, isCreator: true),
        );
      }
      await _enrichMembers(members);
      if (!mounted) return;
      setState(() {
        _members = _sortMembers(members);
        _loading = false;
      });
    } on ApiException catch (e) {
      if (e.statusCode == 404 || e.message.toLowerCase().contains('not found')) {
        try {
          await _loadFromMessages();
        } catch (e2) {
          if (!mounted) return;
          setState(() {
            _err = e2 is ApiException ? e2.message : '$e2';
            _loading = false;
          });
        }
        return;
      }
      if (!mounted) return;
      setState(() {
        _err = e.message;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.55,
      maxChildSize: 0.9,
      minChildSize: 0.35,
      builder: (context, scrollController) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                widget.roomTitle,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
              ),
              Text(
                '${_members.length} member${_members.length == 1 ? '' : 's'}',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                    ),
              ),
              if (_err != null)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                ),
              Expanded(
                child: _loading
                    ? const Center(child: CircularProgressIndicator())
                    : _members.isEmpty
                        ? Center(
                            child: Text(
                              'No members yet.',
                              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                    color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                                  ),
                            ),
                          )
                        : ListView.builder(
                            controller: scrollController,
                            itemCount: _members.length,
                            itemBuilder: (context, i) {
                              final m = _members[i];
                              final name = m['name']?.toString().trim();
                              final email = m['email']?.toString() ?? '';
                              final displayName = (name != null && name.isNotEmpty && name != 'Room creator')
                                  ? name
                                  : (_nameFromEmail(email).isNotEmpty ? _nameFromEmail(email) : 'Member');
                              final isCreator = m['is_creator'] == true;
                              return ListTile(
                                contentPadding: EdgeInsets.zero,
                                leading: UserAvatar(
                                  radius: 22,
                                  imageUrl: m['avatar_url']?.toString(),
                                  name: displayName,
                                ),
                                title: Text(displayName),
                                subtitle: email.isNotEmpty && !isCreator ? Text(email) : null,
                                trailing: isCreator
                                    ? Container(
                                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                        decoration: BoxDecoration(
                                          color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.15),
                                          borderRadius: BorderRadius.circular(12),
                                        ),
                                        child: Text(
                                          'Creator',
                                          style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                                color: LegatoLinkedInTheme.navActiveGold,
                                                fontWeight: FontWeight.w600,
                                              ),
                                        ),
                                      )
                                    : null,
                              );
                            },
                          ),
              ),
            ],
          ),
        );
      },
    );
  }
}
