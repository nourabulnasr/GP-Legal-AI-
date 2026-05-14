import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/screens/history/analysis_detail_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class AnalyzeScreen extends StatefulWidget {
  const AnalyzeScreen({super.key});

  @override
  State<AnalyzeScreen> createState() => _AnalyzeScreenState();
}

class _AnalyzeScreenState extends State<AnalyzeScreen> {
  bool _save = true;
  final _query = TextEditingController();
  bool _busy = false;
  String? _err;

  @override
  void dispose() {
    _query.dispose();
    super.dispose();
  }

  Future<void> _pickAndRun() async {
    final app = context.read<AppServices>();
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['pdf', 'docx', 'png', 'jpg', 'jpeg'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;
    if (!context.mounted) return;
    final f = result.files.single;
    final bytes = f.bytes;
    if (bytes == null || bytes.isEmpty) {
      setState(() => _err = 'Could not read file data. Please try again.');
      return;
    }
    final name = f.name;
    setState(() {
      _busy = true;
      _err = null;
    });
    await WakelockPlus.enable();
    try {
      final data = await app.legato.analyzeContract(
            bytes,
            name,
            useRag: true,
            useMl: true,
            useLlm: true,
            save: _save,
            query: _query.text.trim().isEmpty ? null : _query.text.trim(),
          );
      if (!mounted) return;
      await Navigator.of(context).push<void>(
        MaterialPageRoute<void>(
          builder: (_) => AnalysisDetailScreen(
            title: 'Latest analysis',
            payload: data,
            analysisId: data['analysis_id'] as int?,
          ),
        ),
      );
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      await WakelockPlus.disable();
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final goldBtn = FilledButton.styleFrom(
      backgroundColor: LegatoLinkedInTheme.navActiveGold,
      foregroundColor: const Color(0xFF1B1F23),
    );
    return Scaffold(
      backgroundColor: LegatoLinkedInTheme.background,
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text(
              'Analyze contract',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Text(
              'Upload a PDF, DOCX, or image to check your contract',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary),
            ),
          const SizedBox(height: 16),
          SwitchListTile(
            title: const Text('Save to history'),
            subtitle: const Text('Requires login; backend rejects save without auth'),
            value: _save,
            onChanged: _busy ? null : (v) => setState(() => _save = v),
          ),
          TextField(
            controller: _query,
            decoration: const InputDecoration(
              labelText: 'Optional search query (same as web)',
              border: OutlineInputBorder(),
            ),
            maxLines: 2,
          ),
          const SizedBox(height: 16),
          if (_err != null)
            Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          const SizedBox(height: 8),
            FilledButton.icon(
              style: goldBtn,
              onPressed: _busy ? null : _pickAndRun,
              icon: _busy
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF1B1F23)),
                    )
                  : const Icon(Icons.folder_open),
              label: Text(_busy ? 'Working…' : 'Choose file & analyze'),
            ),
          ],
        ),
      ),
    );
  }
}
