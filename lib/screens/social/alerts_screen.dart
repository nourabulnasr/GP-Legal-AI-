import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

/// Alerts: contract milestones (timeline) + network invites.
class AlertsScreen extends StatefulWidget {
  const AlertsScreen({super.key});

  @override
  State<AlertsScreen> createState() => AlertsScreenState();
}

/// Called by HomeShell whenever the Alerts tab becomes active.
class AlertsScreenState extends State<AlertsScreen> {
  void refresh() => _load();
  bool _loading = true;
  String? _err;
  List<dynamic> _timeline = const [];
  List<dynamic> _invites = const [];

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
    final app = context.read<AppServices>();
    final errs = <String>[];
    var tl = <dynamic>[];
    var inv = <dynamic>[];

    try {
      tl = await app.legato.timelineMe();
    } on ApiException catch (e) {
      // Backend not updated yet -> don't break the whole screen.
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
      _timeline = tl;
      _invites = inv;
      _err = errs.isEmpty ? null : errs.join('\n');
      _loading = false;
    });
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Alerts')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  Text(
                    'Network invitations',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_invites.isEmpty)
                    Text(
                      'No pending invitations.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                    )
                  else
                    ..._invites.map((raw) {
                      final m = Map<String, dynamic>.from(raw as Map);
                      final id = (m['id'] as num?)?.toInt();
                      return Card(
                        child: ListTile(
                          leading: const Icon(Icons.person_add_alt_1_outlined, color: LegatoLinkedInTheme.navActiveGold),
                          title: Text(m['requester_name']?.toString() ?? 'Member'),
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
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                    )
                  else
                    ..._timeline.map((raw) {
                      final m = Map<String, dynamic>.from(raw as Map);
                      return Card(
                        child: ListTile(
                          leading: const Icon(Icons.event_note_outlined, color: LegatoLinkedInTheme.navActiveGold),
                          title: Text(m['label']?.toString() ?? 'Milestone'),
                          subtitle: Text('${m['event_date'] ?? ''} · analysis ${m['analysis_id'] ?? ''}'),
                        ),
                      );
                    }),
                ],
              ),
            ),
    );
  }
}
