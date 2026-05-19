import 'package:flutter/material.dart';

import 'package:legato_mobile/theme/linkedin_theme.dart';

/// OCR chunks with toggle between original and translated text.
class OcrChunkTranslationList extends StatefulWidget {
  const OcrChunkTranslationList({
    super.key,
    required this.chunks,
    required this.targetLang,
  });

  final List<dynamic> chunks;
  final String targetLang;

  @override
  State<OcrChunkTranslationList> createState() => _OcrChunkTranslationListState();
}

class _OcrChunkTranslationListState extends State<OcrChunkTranslationList> {
  bool _showTranslated = true;

  String _original(Map<String, dynamic> c) {
    return (c['normalized_text'] ?? c['text'] ?? '').toString().trim();
  }

  String _translated(Map<String, dynamic> c) {
    final target = widget.targetLang.toLowerCase();
    if (target == 'ar') {
      final ar = (c['translated_ar_text'] ?? '').toString().trim();
      if (ar.isNotEmpty) return ar;
    }
    return (c['translated_text'] ?? c['translated_ar_text'] ?? '').toString().trim();
  }

  @override
  Widget build(BuildContext context) {
    final items = widget.chunks.whereType<Map>().map((e) => Map<String, dynamic>.from(e)).toList();
    if (items.isEmpty) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text('No OCR chunks returned.'),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SegmentedButton<bool>(
          segments: const [
            ButtonSegment(value: false, label: Text('Original'), icon: Icon(Icons.description_outlined)),
            ButtonSegment(value: true, label: Text('Translated'), icon: Icon(Icons.translate)),
          ],
          selected: {_showTranslated},
          onSelectionChanged: (s) => setState(() => _showTranslated = s.first),
        ),
        const SizedBox(height: 12),
        ...items.asMap().entries.map((entry) {
          final i = entry.key;
          final c = entry.value;
          final orig = _original(c);
          final trans = _translated(c);
          final body = _showTranslated && trans.isNotEmpty ? trans : orig;
          final page = c['page'];
          final id = c['id']?.toString();
          final title = page != null ? 'Page $page' : (id ?? 'Chunk ${i + 1}');

          return Card(
            margin: const EdgeInsets.only(bottom: 8),
            child: ExpansionTile(
              initiallyExpanded: i == 0,
              title: Text(title, style: Theme.of(context).textTheme.titleSmall),
              subtitle: Directionality(
                textDirection: (_showTranslated && widget.targetLang.toLowerCase() == 'ar')
                    ? TextDirection.rtl
                    : TextDirection.ltr,
                child: Text(
                  body.isEmpty ? '(empty)' : '${body.length} characters',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                      ),
                ),
              ),
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                  child: Directionality(
                    textDirection: widget.targetLang.toLowerCase() == 'ar'
                        ? TextDirection.rtl
                        : TextDirection.ltr,
                    child: SelectableText(
                      body.isEmpty ? '—' : body,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(height: 1.4),
                      textAlign: widget.targetLang.toLowerCase() == 'ar'
                          ? TextAlign.right
                          : TextAlign.left,
                    ),
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }
}
