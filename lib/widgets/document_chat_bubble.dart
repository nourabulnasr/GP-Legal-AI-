import 'package:flutter/material.dart';

import 'package:legato_mobile/theme/linkedin_theme.dart';

/// One turn in a document chat thread (`POST /chat/document`).
class DocumentChatMessage {
  const DocumentChatMessage({
    required this.role,
    required this.content,
    this.usedFallback,
  });

  final String role;
  final String content;

  /// Set on assistant replies when the API returns `used_fallback`.
  final bool? usedFallback;

  bool get isUser => role == 'user';
}

bool? usedFallbackFromResponse(Map<String, dynamic> res) {
  final v = res['used_fallback'];
  if (v == null) return null;
  if (v is bool) return v;
  if (v is num) return v != 0;
  if (v is String) {
    final s = v.toLowerCase();
    return s == 'true' || s == '1';
  }
  return null;
}

List<Map<String, dynamic>> documentChatHistory(List<DocumentChatMessage> thread) {
  return thread
      .map((m) => <String, dynamic>{'role': m.role, 'content': m.content})
      .toList();
}

/// Full multi-line assistant text plus optional source label (local LFM vs Gemini).
class DocumentChatBubble extends StatelessWidget {
  const DocumentChatBubble({super.key, required this.message});

  final DocumentChatMessage message;

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    final secondary = LegatoLinkedInTheme.textSecondaryAdaptive(context);
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(12),
        constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.88),
        decoration: BoxDecoration(
          color: isUser
              ? LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.22)
              : Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SelectableText(
              message.content,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            if (!isUser && message.usedFallback != null) ...[
              const SizedBox(height: 8),
              Text(
                'Source: ${message.usedFallback! ? 'Gemini fallback' : 'Local LFM'}',
                style: Theme.of(context).textTheme.labelSmall?.copyWith(
                      color: secondary,
                      fontStyle: FontStyle.italic,
                    ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
