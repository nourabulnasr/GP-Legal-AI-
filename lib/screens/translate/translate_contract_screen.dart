import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/screens/translate/ocr_chunk_translation_list.dart';
import 'package:legato_mobile/screens/translate/translation_status_banner.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:legato_mobile/utils/pdf_download.dart';

const _targetLanguages = <String, String>{
  'ar': 'Arabic',
  'en': 'English',
  'fr': 'French',
  'de': 'German',
};

/// Upload contract for OCR + automatic server-side translation (no provider UI).
class TranslateContractScreen extends StatefulWidget {
  const TranslateContractScreen({super.key});

  @override
  State<TranslateContractScreen> createState() => _TranslateContractScreenState();
}

class _TranslateContractScreenState extends State<TranslateContractScreen> {
  String _targetLang = 'ar';
  bool _save = false;
  bool _busy = false;
  bool _cancelled = false;
  String? _err;
  Map<String, dynamic>? _result;

  Future<void> _pickAndTranslate() async {
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
    _cancelled = false;
    setState(() {
      _busy = true;
      _err = null;
      _result = null;
    });
    await WakelockPlus.enable();
    try {
      final data = await app.legato.translateContract(
        bytes,
        f.name,
        translationTargetLang: _targetLang,
        save: _save,
      );
      if (!mounted || _cancelled) return;
      setState(() => _result = data);
    } on ApiException catch (e) {
      if (mounted && !_cancelled) setState(() => _err = e.message);
    } catch (e) {
      if (mounted && !_cancelled) setState(() => _err = e.toString());
    } finally {
      _cancelled = false;
      await WakelockPlus.disable();
      if (mounted) setState(() => _busy = false);
    }
  }

  /// Extracts translated text from chunks using the same key logic as OcrChunkTranslationList.
  String _collectTranslated(List<dynamic> chunks, String targetLang) {
    final lines = <String>[];
    for (final chunk in chunks) {
      if (chunk is! Map) continue;
      final c = Map<String, dynamic>.from(chunk);
      String t = '';
      if (targetLang.toLowerCase() == 'ar') {
        t = c['translated_ar_text']?.toString().trim() ?? '';
      }
      if (t.isEmpty) {
        t = (c['translated_text'] ?? c['translated_ar_text'] ?? '').toString().trim();
      }
      if (t.isNotEmpty) lines.add(t);
    }
    return lines.join('\n\n');
  }

  String? _effectiveSource(Map<String, dynamic>? data) {
    if (data == null) return null;
    final direct = data['source_language_effective']?.toString();
    if (direct != null && direct.isNotEmpty) return direct;
    final ld = data['language_detection'];
    if (ld is Map) return ld['language_code']?.toString();
    return null;
  }

  Future<void> _downloadTranslated(List<dynamic> chunks, String targetLang) async {
    final text = _collectTranslated(chunks, targetLang);
    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No translated text to download.')),
      );
      return;
    }
    final lang = targetLang.toLowerCase();
    final filename = 'translated_contract_$lang.pdf';
    final rtl = lang == 'ar';
    try {
      await downloadPdfFile(filename, text, rtl: rtl);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(kIsWeb ? 'PDF download started' : 'Saved as $filename'),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('PDF download failed: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final goldBtn = FilledButton.styleFrom(
      backgroundColor: LegatoLinkedInTheme.navActiveGold,
      foregroundColor: const Color(0xFF1B1F23),
    );
    final data = _result;
    final tx = data?['translation'] is Map ? Map<String, dynamic>.from(data!['translation'] as Map) : null;
    final chunks = data?['ocr_chunks'] is List ? data!['ocr_chunks'] as List : const <dynamic>[];
    final target = data?['translation_target_lang']?.toString() ?? _targetLang;

    return Scaffold(
      appBar: LegatoAppBar(title: const Text('Translate contract')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text(
              'Translate contract',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Text(
              'Upload a PDF, DOCX, or image. The server translates automatically (Google Cloud, then local LFM).',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                  ),
            ),
            const SizedBox(height: 16),
            DropdownMenu<String>(
              initialSelection: _targetLang,
              label: const Text('Target language'),
              dropdownMenuEntries: [
                for (final e in _targetLanguages.entries)
                  DropdownMenuEntry(value: e.key, label: e.value),
              ],
              onSelected: _busy
                  ? null
                  : (v) {
                      if (v != null) setState(() => _targetLang = v);
                    },
            ),
            const SizedBox(height: 8),
            SwitchListTile(
              title: const Text('Save to history'),
              value: _save,
              onChanged: _busy ? null : (v) => setState(() => _save = v),
            ),
            const SizedBox(height: 8),
            if (_err != null)
              Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            FilledButton.icon(
              style: goldBtn,
              onPressed: _busy ? null : _pickAndTranslate,
              icon: _busy
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF1B1F23)),
                    )
                  : const Icon(Icons.translate),
              label: Text(_busy ? 'Translating…' : 'Choose file & translate'),
            ),
            if (_busy) ...[
              const SizedBox(height: 8),
              OutlinedButton(
                onPressed: () => setState(() {
                  _cancelled = true;
                  _busy = false;
                }),
                child: const Text('Cancel'),
              ),
            ],
            if (data != null) ...[
              const SizedBox(height: 24),
              TranslationStatusBanner(
                translationMeta: tx,
                targetLang: target,
                sourceLang: _effectiveSource(data),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () {
                        final text = _collectTranslated(chunks, target);
                        if (text.isEmpty) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('No translated text to copy.')),
                          );
                          return;
                        }
                        Clipboard.setData(ClipboardData(text: text));
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('Translated text copied to clipboard')),
                        );
                      },
                      icon: const Icon(Icons.copy_outlined, size: 18),
                      label: const Text('Copy'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _downloadTranslated(chunks, target),
                      icon: const Icon(Icons.download_outlined, size: 18),
                      label: const Text('Download PDF'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Text(
                'Document chunks',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              OcrChunkTranslationList(chunks: chunks, targetLang: target),
            ],
          ],
        ),
      ),
    );
  }
}
