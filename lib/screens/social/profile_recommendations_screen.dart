import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class ProfileRecommendationsScreen extends StatefulWidget {
  const ProfileRecommendationsScreen({super.key});

  @override
  State<ProfileRecommendationsScreen> createState() => _ProfileRecommendationsScreenState();
}

class _ProfileRecommendationsScreenState extends State<ProfileRecommendationsScreen> {
  bool _loading = true;
  String? _err;
  List<dynamic> _items = [];

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
      final r = await context.read<AppServices>().legato.getRecommendations(uid);
      if (!mounted) return;
      setState(() {
        _items = (r['items'] as List<dynamic>?) ?? [];
        _loading = false;
      });
    } on ApiException catch (e) {
      if (mounted) setState(() {
        _err = e.message;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Recommendations')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  Text(
                    'Recommendations you received',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_items.isEmpty)
                    Text(
                      'None yet. Ask a colleague to recommend you from their app (write recommendation to your profile).',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                    )
                  else
                    ..._items.map((raw) {
                      final m = raw as Map<String, dynamic>;
                      return Card(
                        child: Padding(
                          padding: const EdgeInsets.all(12),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(m['author_name']?.toString() ?? '', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
                              const SizedBox(height: 6),
                              Text(m['content']?.toString() ?? ''),
                            ],
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
