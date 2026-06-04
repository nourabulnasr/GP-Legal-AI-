import 'package:flutter/material.dart';

import 'package:legato_mobile/screens/share/shared_analysis_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/utils/share_link.dart';

enum ChatDeliveryStatus { none, sent, seen }

class UserChatMessage {
  const UserChatMessage({
    required this.body,
    required this.isMine,
    this.authorName,
    this.createdAt,
    this.status = ChatDeliveryStatus.none,
  });

  final String body;
  final bool isMine;
  final String? authorName;
  final String? createdAt;
  final ChatDeliveryStatus status;
}

String formatChatTime(String? iso) {
  if (iso == null || iso.isEmpty) return '';
  try {
    final dt = DateTime.parse(iso).toLocal();
    final h = dt.hour;
    final m = dt.minute.toString().padLeft(2, '0');
    final hour12 = h == 0 ? 12 : (h > 12 ? h - 12 : h);
    final ampm = h >= 12 ? 'PM' : 'AM';
    return '$hour12:$m $ampm';
  } catch (_) {
    return '';
  }
}

class UserChatBubble extends StatelessWidget {
  const UserChatBubble({
    super.key,
    required this.message,
    this.showAuthor = false,
    this.showMeta = true,
  });

  final UserChatMessage message;
  final bool showAuthor;
  final bool showMeta;

  @override
  Widget build(BuildContext context) {
    final isMine = message.isMine;
    final time = formatChatTime(message.createdAt);
    final shareToken = extractShareTokenFromText(message.body);
    return Align(
      alignment: isMine ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        padding: const EdgeInsets.fromLTRB(12, 10, 12, 8),
        constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.82),
        decoration: BoxDecoration(
          color: isMine
              ? LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.25)
              : Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (showAuthor && !isMine && (message.authorName ?? '').isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Text(
                  message.authorName!,
                  style: Theme.of(context).textTheme.labelSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                        color: LegatoLinkedInTheme.navActiveGold,
                      ),
                ),
              ),
            if (shareToken != null)
              _ShareLinkCard(token: shareToken)
            else
              SelectableText(message.body, style: Theme.of(context).textTheme.bodyMedium),
            if (showMeta && (time.isNotEmpty || (isMine && message.status != ChatDeliveryStatus.none)))
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    if (time.isNotEmpty)
                      Text(
                        time,
                        style: Theme.of(context).textTheme.labelSmall?.copyWith(
                              color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                              fontSize: 11,
                            ),
                      ),
                    if (isMine && message.status != ChatDeliveryStatus.none) ...[
                      const SizedBox(width: 4),
                      Icon(
                        message.status == ChatDeliveryStatus.seen ? Icons.done_all : Icons.done,
                        size: 16,
                        color: message.status == ChatDeliveryStatus.seen
                            ? Colors.lightBlueAccent
                            : LegatoLinkedInTheme.textSecondaryAdaptive(context),
                      ),
                    ],
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _ShareLinkCard extends StatelessWidget {
  const _ShareLinkCard({required this.token});

  final String token;

  void _open(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => SharedAnalysisScreen(token: token),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Material(
      color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.12),
      borderRadius: BorderRadius.circular(10),
      child: InkWell(
        onTap: () => _open(context),
        borderRadius: BorderRadius.circular(10),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
          child: Row(
            children: [
              Icon(Icons.share_outlined, color: LegatoLinkedInTheme.navActiveGold, size: 22),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Shared contract analysis',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Read-only · Tap to view',
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(
                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                          ),
                    ),
                  ],
                ),
              ),
              Icon(
                Icons.chevron_right,
                size: 20,
                color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
