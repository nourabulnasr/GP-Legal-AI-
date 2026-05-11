import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/config/runtime_config.dart';
import 'package:legato_mobile/providers/auth_provider.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _urlCtrl;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _urlCtrl = TextEditingController(text: RuntimeConfig.apiBaseUrl);
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    super.dispose();
  }

  Future<void> _saveUrl() async {
    final raw = _urlCtrl.text.trim();
    if (raw.isEmpty) return;
    setState(() => _saving = true);
    try {
      await RuntimeConfig.setApiBaseUrl(raw);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('API URL updated to $raw — takes effect immediately.'),
          behavior: SnackBarBehavior.floating,
        ),
      );
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _resetUrl() async {
    await RuntimeConfig.clearOverride();
    if (!mounted) return;
    setState(() => _urlCtrl.text = RuntimeConfig.apiBaseUrl);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Reset to compiled default.'),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('API Server', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Text(
            'Override the API base URL for switching between local and production without rebuilding the APK.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.black54),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _urlCtrl,
            decoration: const InputDecoration(
              labelText: 'API Base URL',
              hintText: 'http://192.168.1.x:8000',
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.url,
            autocorrect: false,
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: FilledButton(
                  onPressed: _saving ? null : _saveUrl,
                  child: _saving
                      ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                      : const Text('Save URL'),
                ),
              ),
              const SizedBox(width: 8),
              OutlinedButton(
                onPressed: _saving ? null : _resetUrl,
                child: const Text('Reset'),
              ),
            ],
          ),
          const Divider(height: 32),
          ListTile(
            contentPadding: EdgeInsets.zero,
            leading: const Icon(Icons.logout),
            title: const Text('Sign out'),
            onTap: () async {
              await context.read<AuthProvider>().logout();
              if (context.mounted) Navigator.of(context).popUntil((r) => r.isFirst);
            },
          ),
        ],
      ),
    );
  }
}
