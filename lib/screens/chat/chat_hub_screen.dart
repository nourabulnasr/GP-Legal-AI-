import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/chat/chat_assistant_screen.dart';
import 'package:legato_mobile/screens/chat/chat_analysis_screen.dart';
import 'package:legato_mobile/screens/chat/chat_document_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

class ChatHubScreen extends StatelessWidget {
  const ChatHubScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final isLawyer = context.watch<AuthProvider>().user?.isLawyerAccount ?? false;
    return LegatoPageScaffold(
      title: 'Messaging',
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          children: [
            Text(
              isLawyer
                  ? 'Quick help and navigation from the general assistant.'
                  : 'Assistant uses the cloud API. Document and analysis chat use your local LFM.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
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
            if (!isLawyer) ...[
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
          ],
        ),
      ),
    );
  }
}
