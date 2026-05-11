import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/screens/history/analysis_detail_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  bool _loading = true;
  String? _err;
  List<dynamic> _items = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final list = await context.read<AppServices>().legato.listAnalyses();
      setState(() => _items = list);
    } on ApiException catch (e) {
      final msg = e.statusCode == 401
          ? 'Session expired or not signed in. Pull to refresh after signing in again.'
          : e.message;
      setState(() => _err = msg);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return Scaffold(
        backgroundColor: LegatoLinkedInTheme.background,
        appBar: AppBar(title: const Text('History')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    if (_err != null) {
      return Scaffold(
        backgroundColor: LegatoLinkedInTheme.background,
        appBar: AppBar(title: const Text('History')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(_err!, textAlign: TextAlign.center),
                const SizedBox(height: 12),
                FilledButton(onPressed: _load, child: const Text('Retry')),
              ],
            ),
          ),
        ),
      );
    }
    if (_items.isEmpty) {
      return Scaffold(
        backgroundColor: LegatoLinkedInTheme.background,
        appBar: AppBar(title: const Text('History')),
        body: const Center(child: Text('No saved analyses yet.')),
      );
    }
    return Scaffold(
      backgroundColor: LegatoLinkedInTheme.background,
      appBar: AppBar(title: const Text('History')),
      body: RefreshIndicator(
      onRefresh: _load,
      child: ListView.separated(
        padding: const EdgeInsets.all(12),
        itemCount: _items.length,
        separatorBuilder: (context, i) => const Divider(height: 1),
        itemBuilder: (context, i) {
          final m = _items[i] as Map<String, dynamic>;
          final id = m['id'] as int;
          final name = m['filename']?.toString() ?? 'analysis';
          final created = m['created_at']?.toString() ?? '';
          return Card(
            margin: EdgeInsets.zero,
            child: ListTile(
            title: Text(name),
            subtitle: Text(created),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                IconButton(
                  tooltip: 'Delete',
                  icon: const Icon(Icons.delete_outline),
                  onPressed: () async {
                    final ok = await showDialog<bool>(
                      context: context,
                      builder: (ctx) => AlertDialog(
                        title: const Text('Delete analysis?'),
                        content: Text('Remove "$name" from your history?'),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(ctx, false),
                            child: const Text('Cancel'),
                          ),
                          FilledButton(
                            onPressed: () => Navigator.pop(ctx, true),
                            child: const Text('Delete'),
                          ),
                        ],
                      ),
                    );
                    if (ok != true || !context.mounted) return;
                    try {
                      await context.read<AppServices>().legato.deleteAnalysis(id);
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('Deleted')),
                        );
                        await _load();
                      }
                    } on ApiException catch (e) {
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text(e.message)),
                        );
                      }
                    }
                  },
                ),
                const Icon(Icons.chevron_right),
              ],
            ),
            onTap: () async {
              final nav = Navigator.of(context);
              final messenger = ScaffoldMessenger.of(context);
              final api = context.read<AppServices>().legato;
              showDialog<void>(
                context: context,
                barrierDismissible: false,
                builder: (_) => const Center(child: CircularProgressIndicator()),
              );
              try {
                final detail = await api.getAnalysis(id);
                final raw = detail['result_json'];
                Map<String, dynamic> payload;
                if (raw is String) {
                  payload = Map<String, dynamic>.from(jsonDecode(raw) as Map);
                } else {
                  payload = {};
                }
                if (!mounted) return;
                nav.pop();
                await nav.push<void>(
                  MaterialPageRoute<void>(
                    builder: (_) => AnalysisDetailScreen(
                      title: name,
                      payload: payload,
                      analysisId: id,
                    ),
                  ),
                );
              } on ApiException catch (e) {
                if (mounted) {
                  nav.pop();
                  messenger.showSnackBar(SnackBar(content: Text(e.message)));
                }
              } catch (e) {
                if (mounted) {
                  nav.pop();
                  messenger.showSnackBar(SnackBar(content: Text('$e')));
                }
              }
            },
          ),
          );
        },
      ),
      ),
    );
  }
}
