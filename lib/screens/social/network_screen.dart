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
      final st = await api.getNetworkStats();
      final sug = await api.getNetworkSuggestions();
      final pend = await api.getPendingInvites();
      if (!mounted) return;
      setState(() {
        _stats = st;
        // Handle both 'items' and 'suggestions' response keys.
        _suggestions = (sug['items'] as List<dynamic>?) ??
            (sug['suggestions'] as List<dynamic>?) ??
            <dynamic>[];
        _pending = (pend['items'] as List<dynamic>?) ??
            (pend['invitations'] as List<dynamic>?) ??
            <dynamic>[];
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
      color: LegatoLinkedInTheme.background,
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
                        fillColor: Colors.white,
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
                        onTap: () {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Open a connection from suggestions below.')),
                          );
                        },
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

  void _showPending(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      builder: (ctx) => Material(
        color: LegatoLinkedInTheme.background,
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
      color: Colors.white,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(value, style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            Text(label, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary)),
          ],
        ),
      ),
    );
  }
}
