import 'package:flutter/material.dart';

import 'package:legato_mobile/theme/linkedin_theme.dart';

/// Read-only summary of server-side translation metadata (no provider picker).
class TranslationStatusBanner extends StatelessWidget {
  const TranslationStatusBanner({
    super.key,
    required this.translationMeta,
    this.targetLang,
    this.sourceLang,
  });

  final Map<String, dynamic>? translationMeta;
  final String? targetLang;
  final String? sourceLang;

  static String _labelProvider(String? raw) {
    switch (raw) {
      case 'google_cloud_translate_v2':
        return 'Google Cloud Translation';
      case 'local_lfm_translate':
        return 'Local LFM';
      case 'argos_translate':
        return 'Argos (offline fallback)';
      default:
        return raw ?? '—';
    }
  }

  Color _statusColor(BuildContext context, String? status) {
    switch (status) {
      case 'ok':
        return Colors.green.shade700;
      case 'partial':
        return Colors.orange.shade800;
      case 'skipped':
      case 'disabled':
        return Theme.of(context).colorScheme.error;
      default:
        return LegatoLinkedInTheme.textSecondaryAdaptive(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    final tx = translationMeta ?? const <String, dynamic>{};
    final status = tx['translation_status']?.toString();
    final provider = tx['translation_provider']?.toString();
    final skip = tx['skip_reason']?.toString();
    final effectiveTarget = targetLang ?? tx['translation_target_lang']?.toString() ?? 'ar';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Icon(Icons.translate, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                const SizedBox(width: 8),
                Text('Translation', style: Theme.of(context).textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 12),
            _Row('Status', status ?? '—', valueColor: _statusColor(context, status)),
            _Row('Target language', effectiveTarget.toUpperCase()),
            if (sourceLang != null && sourceLang!.isNotEmpty)
              _Row('Detected source', sourceLang!.toUpperCase()),
            if (provider != null && provider.isNotEmpty)
              _Row('Provider (automatic)', _labelProvider(provider)),
            if (skip != null && skip.isNotEmpty)
              _Row('Note', skip.replaceAll('_', ' ')),
            const SizedBox(height: 8),
            Text(
              'Translation engine is chosen on the server (Google first, then local LFM).',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                  ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Row extends StatelessWidget {
  const _Row(this.label, this.value, {this.valueColor});

  final String label;
  final String value;
  final Color? valueColor;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 130,
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                  ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w500,
                    color: valueColor,
                  ),
            ),
          ),
        ],
      ),
    );
  }
}
