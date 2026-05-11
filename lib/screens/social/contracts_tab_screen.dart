import 'package:flutter/material.dart';

import 'package:legato_mobile/screens/analyze/analyze_screen.dart';
import 'package:legato_mobile/screens/features/features_hub_screen.dart';
import 'package:legato_mobile/screens/history/history_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

/// Contracts tab: quick entry to analysis workflow + full tools grid (12 tools).
class ContractsTabScreen extends StatelessWidget {
  const ContractsTabScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: LegatoLinkedInTheme.background,
      child: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text(
              'Contracts',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Text(
              'Analyze documents, review saved analyses, and open all Legato tools.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary),
            ),
            const SizedBox(height: 20),
            Card(
              child: ListTile(
                leading: Icon(Icons.upload_file_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: const Text('Analyze contract'),
                subtitle: const Text('Upload PDF / DOCX / image — OCR + rules + RAG'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const AnalyzeScreen()),
                ),
              ),
            ),
            Card(
              child: ListTile(
                leading: Icon(Icons.history, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: const Text('History'),
                subtitle: const Text('Saved analyses on your account'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const HistoryScreen()),
                ),
              ),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: LegatoLinkedInTheme.navActiveGold,
                foregroundColor: const Color(0xFF1B1F23),
              ),
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => const FeaturesHubScreen()),
              ),
              icon: const Icon(Icons.apps_outlined),
              label: const Text('Open all tools (12)'),
            ),
          ],
        ),
      ),
    );
  }
}
