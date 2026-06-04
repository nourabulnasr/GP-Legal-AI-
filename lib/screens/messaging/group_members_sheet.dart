import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/user_avatar.dart';

Future<void> showGroupMembersSheet(
  BuildContext context, {
  required int conversationId,
  required String groupTitle,
  required int? createdBy,
}) {
  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    builder: (ctx) => _GroupMembersSheet(
      conversationId: conversationId,
      groupTitle: groupTitle,
      createdBy: createdBy,
    ),
  );
}

class _GroupMembersSheet extends StatefulWidget {
  const _GroupMembersSheet({
    required this.conversationId,
    required this.groupTitle,
    required this.createdBy,
  });

  final int conversationId;
  final String groupTitle;
  final int? createdBy;

  @override
  State<_GroupMembersSheet> createState() => _GroupMembersSheetState();
}

class _GroupMembersSheetState extends State<_GroupMembersSheet> {
  bool _loading = true;
  bool _adding = false;
  String? _err;
  List<dynamic> _members = [];

  bool get _isCreator {
    final me = context.read<AuthProvider>().user?.id;
    return me != null && widget.createdBy == me;
  }

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final conv = await context.read<AppServices>().legato.getConversation(widget.conversationId);
      if (!mounted) return;
      setState(() {
        _members = (conv['members'] as List<dynamic>?) ?? [];
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

  Future<void> _addMembers() async {
    final connections = await context.read<AppServices>().legato.getNetworkConnections();
    final items = (connections['items'] as List<dynamic>?) ?? [];
    final existing = _members.map((m) => (m as Map)['user_id']).whereType<num>().map((n) => n.toInt()).toSet();
    final candidates = items.where((raw) {
      if (raw is! Map) return false;
      final id = (raw['user_id'] as num?)?.toInt();
      return id != null && !existing.contains(id);
    }).toList();
    if (candidates.isEmpty) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('All your connections are already in this group.')),
      );
      return;
    }

    final selected = <int>{};
    if (!mounted) return;
    final picked = await showDialog<Set<int>>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDlg) => AlertDialog(
          title: const Text('Add members'),
          content: SizedBox(
            width: double.maxFinite,
            child: ListView(
              shrinkWrap: true,
              children: [
                for (final raw in candidates)
                  if (raw is Map)
                    CheckboxListTile(
                      value: selected.contains((raw['user_id'] as num?)?.toInt()),
                      onChanged: (v) {
                        final id = (raw['user_id'] as num?)?.toInt();
                        if (id == null) return;
                        setDlg(() {
                          if (v == true) {
                            selected.add(id);
                          } else {
                            selected.remove(id);
                          }
                        });
                      },
                      title: Text(raw['display_name']?.toString() ?? raw['name']?.toString() ?? raw['email']?.toString() ?? 'Member'),
                    ),
              ],
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
            FilledButton(
              onPressed: selected.isEmpty ? null : () => Navigator.pop(ctx, selected),
              child: const Text('Add'),
            ),
          ],
        ),
      ),
    );
    if (picked == null || picked.isEmpty || !mounted) return;
    setState(() => _adding = true);
    try {
      await context.read<AppServices>().legato.addGroupMembers(widget.conversationId, picked.toList());
      if (!mounted) return;
      await _load();
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Members added')));
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    } finally {
      if (mounted) setState(() => _adding = false);
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
                widget.groupTitle,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
              ),
              Text(
                '${_members.length} members',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                    ),
              ),
              if (_isCreator)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  child: FilledButton.icon(
                    onPressed: _adding ? null : _addMembers,
                    icon: _adding
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.person_add_outlined),
                    label: const Text('Add connection'),
                  ),
                ),
              if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
              Expanded(
                child: _loading
                    ? const Center(child: CircularProgressIndicator())
                    : ListView.builder(
                        controller: scrollController,
                        itemCount: _members.length,
                        itemBuilder: (context, i) {
                          final m = Map<String, dynamic>.from(_members[i] as Map);
                          final name = m['name']?.toString() ?? m['email']?.toString() ?? 'Member';
                          final isCreator = (m['user_id'] as num?)?.toInt() == widget.createdBy;
                          return ListTile(
                            leading: UserAvatar(
                              radius: 20,
                              imageUrl: m['avatar_url']?.toString(),
                              name: name,
                            ),
                            title: Text(name),
                            subtitle: Text(m['email']?.toString() ?? ''),
                            trailing: isCreator
                                ? Text(
                                    'Creator',
                                    style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                          color: LegatoLinkedInTheme.navActiveGold,
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
