import 'package:flutter/material.dart';

import 'package:legato_mobile/screens/chat/chat_assistant_screen.dart';
import 'package:legato_mobile/screens/chat/chat_analysis_screen.dart';
import 'package:legato_mobile/screens/chat/chat_document_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class ChatHubScreen extends StatelessWidget {
  const ChatHubScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: LegatoLinkedInTheme.background,
      child: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          children: [
            Text(
              'Messaging',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 4),
            Text(
              'Assistant uses the cloud API. Document and analysis chat use your local LFM.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary),
            ),
            const SizedBox(height: 20),
            Card(
              child: ListTile(
                leading: Icon(Icons.smart_toy_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: const Text('General assistant'),
                subtitle: const Text('Quick help and navigation'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const ChatAssistantScreen()),
                ),
              ),
            ),
            Card(
              child: ListTile(
                leading: Icon(Icons.description_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: const Text('Chat with document'),
                subtitle: const Text('LFM + uploaded or saved analysis'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const ChatDocumentScreen()),
                ),
              ),
            ),
            Card(
              child: ListTile(
                leading: Icon(Icons.forum_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: const Text('Chat about saved analysis'),
                subtitle: const Text('LFM with rule hits and OCR context'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const ChatAnalysisScreen()),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
