import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/admin/admin_screen.dart';
import 'package:legato_mobile/screens/lawyer/lawyer_application_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:legato_mobile/widgets/user_avatar.dart';

String _relativeTime(String iso) {
  try {
    final dt = DateTime.parse(iso).toLocal();
    final d = DateTime.now().difference(dt);
    if (d.inSeconds < 60) return '${d.inSeconds}s ago';
    if (d.inMinutes < 60) return '${d.inMinutes}m ago';
    if (d.inHours < 48) return '${d.inHours}h ago';
    if (d.inDays < 14) return '${d.inDays}d ago';
    return '${dt.day}/${dt.month}/${dt.year}';
  } catch (_) {
    return iso;
  }
}

/// Alerts: feed activity + contract milestones + network invites.
class AlertsScreen extends StatefulWidget {
  const AlertsScreen({super.key, this.onOpenPost});

  final void Function(int postId)? onOpenPost;

  @override
  State<AlertsScreen> createState() => AlertsScreenState();
}

/// Called by HomeShell whenever the Alerts tab becomes active.
class AlertsScreenState extends State<AlertsScreen> {
  void refresh() => _load();
  bool _loading = false;
  String? _err;
  List<dynamic> _activity = const [];
  List<dynamic> _timeline = const [];
  List<dynamic> _invites = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    if (_loading) return;
    setState(() {
      _loading = true;
      _err = null;
    });
    final app = context.read<AppServices>();
    final errs = <String>[];
    var activity = <dynamic>[];
    var tl = <dynamic>[];
    var inv = <dynamic>[];

    try {
      final r = await app.legato.listNotifications();
      activity = (r['items'] as List<dynamic>?) ?? [];
    } on ApiException catch (e) {
      if (e.statusCode != 404) errs.add('Activity: ${e.message}');
    } catch (e) {
      errs.add('Activity: $e');
    }

    try {
      tl = await app.legato.timelineMe();
    } on ApiException catch (e) {
      if (e.statusCode != 404) errs.add('Milestones: ${e.message}');
    } catch (e) {
      errs.add('Milestones: $e');
    }

    try {
      final r = await app.legato.getPendingInvites();
      inv = (r['items'] as List<dynamic>?) ?? [];
    } on ApiException catch (e) {
      if (e.statusCode != 404) errs.add('Network: ${e.message}');
    } catch (e) {
      errs.add('Network: $e');
    }

    if (!mounted) return;
    setState(() {
      _activity = activity;
      _timeline = tl;
      _invites = inv;
      _err = errs.isEmpty ? null : errs.join('\n');
      _loading = false;
    });
  }

  IconData _iconForType(String type) {
    switch (type) {
      case 'like':
        return Icons.favorite_outline;
      case 'comment':
        return Icons.chat_bubble_outline;
      case 'connection_post':
        return Icons.article_outlined;
      case 'lawyer_application':
        return Icons.gavel_outlined;
      case 'lawyer_review':
        return Icons.verified_outlined;
      default:
        return Icons.notifications_outlined;
    }
  }

  int? _postIdFromNotification(Map<String, dynamic> m) {
    final raw = m['post_id'] ?? m['postId'];
    if (raw is num) return raw.toInt();
    if (raw is String) return int.tryParse(raw.trim());
    return null;
  }

  Future<void> _onActivityTap(Map<String, dynamic> m) async {
    final id = (m['id'] as num?)?.toInt();
    final type = m['type']?.toString() ?? '';
    final postId = _postIdFromNotification(m);
    if (id != null) {
      try {
        await context.read<AppServices>().legato.markNotificationRead(id);
      } catch (_) {}
    }
    if (!mounted) return;
    setState(() {
      _activity = _activity.map((raw) {
        if (raw is! Map) return raw;
        final copy = Map<String, dynamic>.from(raw);
        if ((copy['id'] as num?)?.toInt() == id) copy['read'] = true;
        return copy;
      }).toList();
    });
    // Lawyer review → lawyer sees the admin's decision.
    if (type == 'lawyer_review') {
      await Navigator.of(context).push(
        MaterialPageRoute<void>(builder: (_) => const LawyerApplicationScreen()),
      );
      return;
    }
    // Lawyer application → admin is taken straight to the Lawyers tab.
    if (type == 'lawyer_application') {
      final isAdmin = context.read<AuthProvider>().user?.isAdmin ?? false;
      if (isAdmin) {
        await Navigator.of(context).push(
          MaterialPageRoute<void>(builder: (_) => const AdminScreen(initialTab: 2)),
        );
      }
      return;
    }
    if (postId == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('This alert is not linked to a post.')),
      );
      return;
    }
    if (widget.onOpenPost != null) {
      widget.onOpenPost!(postId);
    }
  }

  Future<void> _acceptInvite(int id) async {
    try {
      await context.read<AppServices>().legato.acceptNetworkInvite(id);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Invitation accepted')));
      await _load();
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  Future<void> _deleteActivity(int id) async {
    try {
      await context.read<AppServices>().legato.deleteNotification(id);
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      return;
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
      return;
    }
    if (!mounted) return;
    setState(() {
      _activity = _activity.where((raw) {
        if (raw is! Map) return true;
        return (raw['id'] as num?)?.toInt() != id;
      }).toList();
    });
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Alert deleted')),
    );
  }

  Future<void> _deleteMilestone(int id, String label) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete milestone?'),
        content: Text('Remove "$label" from your timeline?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    try {
      await context.read<AppServices>().legato.deleteTimelineEvent(id);
      if (!mounted) return;
      setState(() {
        _timeline = _timeline.where((raw) {
          if (raw is! Map) return true;
          return (raw['id'] as num?)?.toInt() != id;
        }).toList();
      });
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Milestone deleted')));
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: LegatoAppBar(
        title: const Text('Alerts'),
        actions: [
          if (_activity.any((a) => a is Map && a['read'] != true))
            TextButton(
              onPressed: () async {
                try {
                  await context.read<AppServices>().legato.markAllNotificationsRead();
                  if (mounted) await _load();
                } catch (_) {}
              },
              child: const Text('Mark all read'),
            ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  Text(
                    'Feed activity',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_activity.isEmpty)
                    Text(
                      'No activity yet. Likes, comments, and posts from connections appear here.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          ),
                    )
                  else
                    ..._activity.map((raw) {
                      final m = Map<String, dynamic>.from(raw as Map);
                      final id = (m['id'] as num?)?.toInt();
                      final unread = m['read'] != true;
                      return Card(
                        color: unread
                            ? LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.08)
                            : null,
                        child: ListTile(
                          leading: UserAvatar(
                            radius: 22,
                            imageUrl: m['actor_avatar_url']?.toString(),
                            name: m['actor_name']?.toString() ?? 'Member',
                          ),
                          title: Row(
                            children: [
                              if (m['actor_is_verified_lawyer'] == true) ...[
                                const Tooltip(message: 'Verified Lawyer', child: Icon(Icons.verified, size: 14, color: Color(0xFF0A66C2))),
                                const SizedBox(width: 4),
                              ],
                              Flexible(
                                child: Text(
                                  m['message']?.toString() ?? 'Activity',
                                  style: TextStyle(fontWeight: unread ? FontWeight.w600 : FontWeight.normal),
                                ),
                              ),
                            ],
                          ),
                          subtitle: Text(_relativeTime(m['created_at']?.toString() ?? '')),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                _iconForType(m['type']?.toString() ?? ''),
                                size: 18,
                                color: LegatoLinkedInTheme.navActiveGold,
                              ),
                              if (id != null)
                                IconButton(
                                  tooltip: 'Delete alert',
                                  icon: Icon(Icons.delete_outline, color: Theme.of(context).colorScheme.error),
                                  onPressed: () => _deleteActivity(id),
                                ),
                              if (_postIdFromNotification(m) != null) const Icon(Icons.chevron_right),
                            ],
                          ),
                          onTap: () => _onActivityTap(m),
                        ),
                      );
                    }),
                  const SizedBox(height: 16),
                  Text(
                    'Network invitations',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_invites.isEmpty)
                    Text(
                      'No pending invitations.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          ),
                    )
                  else
                    ..._invites.map((raw) {
                      final m = Map<String, dynamic>.from(raw as Map);
                      final id = (m['id'] as num?)?.toInt();
                      return Card(
                        child: ListTile(
                          leading: const Icon(Icons.person_add_alt_1_outlined, color: LegatoLinkedInTheme.navActiveGold),
                          title: Row(
                            children: [
                              Flexible(child: Text(m['requester_name']?.toString() ?? 'Member', overflow: TextOverflow.ellipsis)),
                              if (m['requester_is_verified_lawyer'] == true) ...[
                                const SizedBox(width: 4),
                                const Tooltip(message: 'Verified Lawyer', child: Icon(Icons.verified, size: 14, color: Color(0xFF0A66C2))),
                              ],
                            ],
                          ),
                          subtitle: Text(m['created_at']?.toString() ?? ''),
                          trailing: TextButton(
                            onPressed: id == null ? null : () => _acceptInvite(id),
                            child: const Text('Accept'),
                          ),
                        ),
                      );
                    }),
                  const SizedBox(height: 16),
                  Text(
                    'Contract milestones',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_timeline.isEmpty)
                    Text(
                      'No milestones yet.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          ),
                    )
                  else
                    ..._timeline.map((raw) {
                      final m = Map<String, dynamic>.from(raw as Map);
                      final id = (m['id'] as num?)?.toInt();
                      final label = m['label']?.toString() ?? 'Milestone';
                      return Card(
                        child: ListTile(
                          leading: const Icon(Icons.event_note_outlined, color: LegatoLinkedInTheme.navActiveGold),
                          title: Text(label),
                          subtitle: Text('${m['event_date'] ?? ''} | analysis ${m['analysis_id'] ?? ''}'),
                          trailing: id == null
                              ? null
                              : IconButton(
                                  tooltip: 'Delete milestone',
                                  icon: Icon(Icons.delete_outline, color: Theme.of(context).colorScheme.error),
                                  onPressed: () => _deleteMilestone(id, label),
                                ),
                        ),
                      );
                    }),
                ],
              ),
            ),
    );
  }
}
