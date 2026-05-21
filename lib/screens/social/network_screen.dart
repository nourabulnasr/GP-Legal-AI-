import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/screens/social/member_profile_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class NetworkScreen extends StatefulWidget {
  const NetworkScreen({super.key});

  @override
  State<NetworkScreen> createState() => _NetworkScreenState();
}

class _NetworkScreenState extends State<NetworkScreen> {
  final _search = TextEditingController();
  bool _loading = true;
  String? _err;
  Map<String, dynamic>? _stats;
  List<dynamic> _suggestions = [];
  List<dynamic> _pending = [];
  List<dynamic> _connections = [];
  bool _searching = false;
  List<dynamic> _results = [];
  final Set<int> _sentInvites = {};

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final api = context.read<AppServices>().legato;
      final results = await Future.wait<dynamic>([
        api.getNetworkStats(),
        api.getNetworkSuggestions(),
        api.getPendingInvites(),
        _loadConnections(api),
      ]);
      if (!mounted) return;
      final st = results[0] as Map<String, dynamic>;
      final sug = results[1] as Map<String, dynamic>;
      final pend = results[2] as Map<String, dynamic>;
      final conn = results[3] as List<dynamic>;
      setState(() {
        _stats = st;
        // Handle both 'items' and 'suggestions' response keys.
        _suggestions = (sug['items'] as List<dynamic>?) ??
            (sug['suggestions'] as List<dynamic>?) ??
            <dynamic>[];
        _pending = (pend['items'] as List<dynamic>?) ??
            (pend['invitations'] as List<dynamic>?) ??
            <dynamic>[];
        _connections = conn;
        _loading = false;
      });
    } on ApiException catch (e) {
      if (mounted) {
        setState(() {
          _err = e.message;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _err = e.toString();
          _loading = false;
        });
      }
    }
  }

  Future<List<dynamic>> _loadConnections(dynamic api) async {
    try {
      final connRes = await api.getNetworkConnections();
      return (connRes['items'] as List<dynamic>?) ??
          (connRes['connections'] as List<dynamic>?) ??
          <dynamic>[];
    } on ApiException catch (e) {
      if (e.statusCode != 404) rethrow; // 404 = endpoint not yet deployed, silently ignore
      return <dynamic>[];
    } catch (_) {
      return <dynamic>[];
    }
  }

  Future<void> _runSearch(String q) async {
    final query = q.trim();
    if (query.isEmpty) {
      if (mounted) setState(() => _results = []);
      return;
    }
    setState(() {
      _searching = true;
      _err = null;
    });
    try {
      final r = await context.read<AppServices>().legato.searchNetwork(query);
      if (!mounted) return;
      setState(() {
        _results = (r['items'] as List<dynamic>?) ?? [];
        _searching = false;
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.message;
        _searching = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.toString();
        _searching = false;
      });
    }
  }

  Future<void> _invite(int userId) async {
    try {
      await context.read<AppServices>().legato.sendNetworkInvite(userId);
      if (!mounted) return;
      setState(() => _sentInvites.add(userId));
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Invitation sent — they will see your request')),
      );
      await _load();
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: Theme.of(context).scaffoldBackgroundColor,
      child: SafeArea(
        child: RefreshIndicator(
          onRefresh: _load,
          child: _loading
              ? const Center(child: CircularProgressIndicator())
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    Text(
                      'My Network',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _search,
                      decoration: InputDecoration(
                        hintText: 'Search lawyers, firms, or specialties…',
                        prefixIcon: const Icon(Icons.search),
                        filled: true,
                        fillColor: Theme.of(context).colorScheme.surface,
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      textInputAction: TextInputAction.search,
                      onSubmitted: _runSearch,
                    ),
                    if (_searching) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator(minHeight: 2)),
                    if (_err != null) ...[
                      const SizedBox(height: 12),
                      Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                    ],
                    if (_results.isNotEmpty) ...[
                      const SizedBox(height: 12),
                      Text(
                        'Results',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 8),
                      ..._results.map((raw) {
                        final m = Map<String, dynamic>.from(raw as Map);
                        final uid = (m['user_id'] as num?)?.toInt() ?? 0;
                        final sent = _sentInvites.contains(uid);
                        return Card(
                          child: ListTile(
                            leading: const CircleAvatar(child: Icon(Icons.person_outline)),
                            title: Text(m['name']?.toString() ?? 'Member'),
                            subtitle: Text(
                              '${m['subtitle'] ?? ''}\n${m['location'] ?? ''}'.trim(),
                              maxLines: 2,
                            ),
                            onTap: uid > 0
                                ? () => Navigator.of(context).push(
                                      MaterialPageRoute<void>(builder: (_) => MemberProfileScreen(userId: uid)),
                                    )
                                : null,
                            trailing: sent
                                ? const Chip(label: Text('Sent ✓'))
                                : FilledButton.tonal(
                                    onPressed: uid > 0 ? () => _invite(uid) : null,
                                    child: const Text('Connect'),
                                  ),
                          ),
                        );
                      }),
                      const SizedBox(height: 12),
                    ],
                    const SizedBox(height: 16),
                    if (_stats != null) _StatsGrid(stats: _stats!),
                    const SizedBox(height: 16),
                    Card(
                      child: ListTile(
                        leading: Icon(Icons.people_outline, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                        title: const Text('Connections'),
                        subtitle: const Text('Manage your network'),
                        trailing: const Icon(Icons.chevron_right),
                        onTap: () => _showConnectionsSheet(context),
                      ),
                    ),
                    if (_pending.isNotEmpty)
                      Card(
                        child: ListTile(
                          leading: Badge(
                            label: Text('${_pending.length}'),
                            child: const Icon(Icons.mail_outline),
                          ),
                          title: const Text('Invitations'),
                          subtitle: Text('${_pending.length} pending requests'),
                          trailing: const Icon(Icons.chevron_right),
                          onTap: () => _showPending(context),
                        ),
                      ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'People you may know',
                          style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                        ),
                        TextButton(onPressed: _load, child: const Text('Refresh')),
                      ],
                    ),
                    ..._suggestions.map((raw) {
                      final m = Map<String, dynamic>.from(raw as Map);
                      final uid = (m['user_id'] as num?)?.toInt() ?? 0;
                      final sent = _sentInvites.contains(uid);
                      return Card(
                        child: ListTile(
                          leading: const CircleAvatar(child: Icon(Icons.person_outline)),
                          title: Text(m['name']?.toString() ?? 'Member'),
                          subtitle: Text(
                            '${m['subtitle'] ?? ''}\n${m['location'] ?? ''}'.trim(),
                            maxLines: 2,
                          ),
                          onTap: uid > 0
                              ? () => Navigator.of(context).push(
                                    MaterialPageRoute<void>(builder: (_) => MemberProfileScreen(userId: uid)),
                                  )
                              : null,
                          trailing: sent
                              ? const Chip(label: Text('Sent ✓'))
                              : FilledButton.tonal(
                                  onPressed: uid > 0 ? () => _invite(uid) : null,
                                  child: const Text('Connect'),
                                ),
                        ),
                      );
                    }),
                  ],
                ),
        ),
      ),
    );
  }

  void _showConnectionsSheet(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.95,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        builder: (ctx, scrollCtrl) => Material(
          color: Theme.of(context).scaffoldBackgroundColor,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
          child: ListView(
            controller: scrollCtrl,
            padding: const EdgeInsets.all(16),
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 16),
                  decoration: BoxDecoration(
                    color: LegatoLinkedInTheme.textSecondaryAdaptive(context).withValues(alpha: 0.4),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const Text('My Network', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
              const SizedBox(height: 16),
              if (_connections.isNotEmpty) ...[
                Text(
                  'Active connections (${_connections.length})',
                  style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
                ),
                const SizedBox(height: 8),
                for (final raw in _connections)
                  Builder(builder: (ctx2) {
                    final m = Map<String, dynamic>.from(raw as Map);
                    final uid = (m['user_id'] as num?)?.toInt() ?? 0;
                    return ListTile(
                      leading: const CircleAvatar(child: Icon(Icons.person_outline)),
                      title: Text(m['name']?.toString() ?? m['display_name']?.toString() ?? 'Member'),
                      subtitle: Text(m['subtitle']?.toString() ?? m['title']?.toString() ?? ''),
                      onTap: uid > 0
                          ? () => Navigator.of(context).push(
                                MaterialPageRoute<void>(
                                  builder: (_) => MemberProfileScreen(userId: uid),
                                ),
                              )
                          : null,
                    );
                  }),
                const Divider(height: 24),
              ] else if (_stats != null) ...[
                Text(
                  '${_stats!['connections_count'] ?? _stats!['total_connections'] ?? _stats!['connections'] ?? 0} connections',
                  style: const TextStyle(fontSize: 14),
                ),
                const Divider(height: 24),
              ],
              if (_pending.isNotEmpty) ...[
                Text(
                  'Pending invitations (${_pending.length})',
                  style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
                ),
                const SizedBox(height: 8),
                for (final raw in _pending)
                  _PendingTile(
                    raw: raw,
                    onAccept: (id) async {
                      try {
                        await context.read<AppServices>().legato.acceptNetworkInvite(id);
                        if (ctx.mounted) Navigator.pop(ctx);
                        await _load();
                      } on ApiException catch (e) {
                        if (context.mounted) {
                          ScaffoldMessenger.of(context)
                              .showSnackBar(SnackBar(content: Text(e.message)));
                        }
                      }
                    },
                  ),
                const Divider(height: 24),
              ],
              const Text(
                'People you may know',
                style: TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
              ),
              const SizedBox(height: 8),
              for (final raw in _suggestions)
                Builder(builder: (ctx2) {
                  final m = Map<String, dynamic>.from(raw as Map);
                  final uid = (m['user_id'] as num?)?.toInt() ?? 0;
                  final sent = _sentInvites.contains(uid);
                  return ListTile(
                    leading: const CircleAvatar(child: Icon(Icons.person_outline)),
                    title: Text(m['name']?.toString() ?? 'Member'),
                    subtitle: Text(m['subtitle']?.toString() ?? ''),
                    trailing: sent
                        ? const Chip(label: Text('Sent ✓'))
                        : FilledButton.tonal(
                            onPressed: uid > 0 ? () => _invite(uid) : null,
                            child: const Text('Connect'),
                          ),
                  );
                }),
            ],
          ),
        ),
      ),
    );
  }

  void _showPending(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      builder: (ctx) => Material(
        color: Theme.of(context).scaffoldBackgroundColor,
        child: SafeArea(
          child: ListView(
            children: [
              const Padding(
                padding: EdgeInsets.all(16),
                child: Text('Pending invitations', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
              ),
              for (final raw in _pending)
                _PendingTile(
                  raw: raw,
                  onAccept: (id) async {
                    try {
                      await context.read<AppServices>().legato.acceptNetworkInvite(id);
                      if (ctx.mounted) Navigator.pop(ctx);
                      await _load();
                    } on ApiException catch (e) {
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
                      }
                    }
                  },
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PendingTile extends StatelessWidget {
  const _PendingTile({required this.raw, required this.onAccept});

  final dynamic raw;
  final Future<void> Function(int id) onAccept;

  @override
  Widget build(BuildContext context) {
    final m = Map<String, dynamic>.from(raw as Map);
    final id = (m['id'] as num?)?.toInt();
    return ListTile(
      title: Text(m['requester_name']?.toString() ?? ''),
      trailing: TextButton(
        onPressed: id != null ? () => onAccept(id) : null,
        child: const Text('Accept'),
      ),
    );
  }
}

class _StatsGrid extends StatelessWidget {
  const _StatsGrid({required this.stats});

  final Map<String, dynamic> stats;

  @override
  Widget build(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      mainAxisSpacing: 10,
      crossAxisSpacing: 10,
      childAspectRatio: 1.6,
      children: [
        _statCard(context, '${stats['connections'] ?? 0}', 'Connections'),
        _statCard(context, '${stats['endorsements'] ?? 0}', 'Endorsements'),
        _statCard(context, '${stats['profile_views'] ?? 0}', 'Who viewed you'),
        _statCard(context, '${stats['invitations_pending'] ?? 0}', 'Invitations'),
      ],
    );
  }

  Widget _statCard(BuildContext context, String value, String label) {
    return Material(
      color: Theme.of(context).colorScheme.surface,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(value, style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            Text(label, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context))),
          ],
        ),
      ),
    );
  }
}
