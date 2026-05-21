import 'dart:async';

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
  String _statusText = '';
  Timer? _progressTimer;
  String? _err;

  @override
  void dispose() {
    _progressTimer?.cancel();
    _query.dispose();
    super.dispose();
  }

  void _startProgressTimer() {
    _progressTimer?.cancel();
    _progressTimer = Timer(const Duration(seconds: 5), () {
      if (mounted && _busy) setState(() => _statusText = 'Extracting text…');
      _progressTimer = Timer(const Duration(seconds: 13), () {
        if (mounted && _busy) setState(() => _statusText = 'Running analysis…');
      });
    });
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
      _statusText = 'Uploading document…';
    });
    _startProgressTimer();
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
      if (data['outdated_law_detected'] == true) {
        await showDialog<void>(
          context: context,
          barrierDismissible: false,
          builder: (ctx) => AlertDialog(
            title: const Text('Outdated Law Detected'),
            content: const SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'هذا العقد يستند إلى قانون العمل رقم 12 لسنة 2003 الذي تم استبداله بقانون العمل رقم 14 لسنة 2025. يجب مراجعة هذا العقد وتحديثه.',
                    textDirection: TextDirection.rtl,
                    textAlign: TextAlign.right,
                  ),
                  SizedBox(height: 12),
                  Text(
                    'This contract references Labor Law No. 12 of 2003 which has been replaced by Egyptian Labor Law No. 14 of 2025. This contract needs to be reviewed and updated.',
                  ),
                ],
              ),
            ),
            actions: [
              FilledButton(
                style: FilledButton.styleFrom(
                  backgroundColor: const Color(0xFFC9A227),
                  foregroundColor: const Color(0xFF1B1F23),
                ),
                onPressed: () => Navigator.of(ctx).pop(),
                child: const Text('Understood'),
              ),
            ],
          ),
        );
      }
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
      _progressTimer?.cancel();
      _progressTimer = null;
      await WakelockPlus.disable();
      if (mounted) setState(() { _busy = false; _statusText = ''; });
    }
  }

  @override
  Widget build(BuildContext context) {
    final goldBtn = FilledButton.styleFrom(
      backgroundColor: LegatoLinkedInTheme.navActiveGold,
      foregroundColor: const Color(0xFF1B1F23),
    );
    return Scaffold(
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
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
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
              label: Text(_busy ? _statusText : 'Choose file & analyze'),
            ),
          ],
        ),
      ),
    );
  }
}
