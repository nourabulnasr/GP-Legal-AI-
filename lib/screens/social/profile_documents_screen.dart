import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/history/analysis_detail_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

class ProfileDocumentsScreen extends StatefulWidget {
  const ProfileDocumentsScreen({super.key});

  @override
  State<ProfileDocumentsScreen> createState() => _ProfileDocumentsScreenState();
}

class _ProfileDocumentsScreenState extends State<ProfileDocumentsScreen> {
  bool _loading = true;
  String? _err;
  List<dynamic> _analyses = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final uid = context.read<AuthProvider>().user?.id;
    if (uid == null) return;
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final analyses = await context.read<AppServices>().legato.listAnalyses();
      if (!mounted) return;
      setState(() {
        _analyses = analyses;
        _loading = false;
      });
    } on ApiException catch (e) {
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

  Future<void> _openAnalysis(Map<String, dynamic> m) async {
    final id = (m['id'] as num?)?.toInt();
    if (id == null) return;
    final name = m['filename']?.toString() ?? 'Contract';
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
      final Map<String, dynamic> payload;
      if (raw is String) {
        payload = Map<String, dynamic>.from(jsonDecode(raw) as Map);
      } else if (raw is Map) {
        payload = Map<String, dynamic>.from(raw);
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
      if (!mounted) return;
      nav.pop();
      messenger.showSnackBar(SnackBar(content: Text(e.message)));
    } catch (e) {
      if (!mounted) return;
      nav.pop();
      messenger.showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: LegatoAppBar(title: const Text('My Documents')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  Text(
                    'Analyzed contracts (your uploads)',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_analyses.isEmpty)
                    Text(
                      'No saved analyses yet. Use Analyze contract from Home.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          ),
                    )
                  else
                    ..._analyses.map((raw) {
                      final m = Map<String, dynamic>.from(raw as Map);
                      return Card(
                        child: ListTile(
                          leading: const Icon(Icons.description_outlined, color: LegatoLinkedInTheme.navActiveGold),
                          title: Text(m['filename']?.toString() ?? 'Contract'),
                          subtitle: Text(
                            m['created_at']?.toString() ?? '',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          trailing: const Icon(Icons.chevron_right),
                          onTap: () => _openAnalysis(m),
                        ),
                      );
                    }),
                ],
              ),
            ),
    );
  }
}
