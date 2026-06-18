import 'package:flutter/material.dart';

import 'package:legato_mobile/screens/chat/chat_assistant_screen.dart';
import 'package:legato_mobile/screens/features/phase5_screens.dart';
import 'package:legato_mobile/screens/translate/translate_contract_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

/// Entry to all Phase-5 capabilities (wired to `/legato/*` API).
class FeaturesHubScreen extends StatelessWidget {
  const FeaturesHubScreen({super.key});

  static int get toolCount => _items.length;
  static String get openAllToolsLabel => 'Open all tools ($toolCount)';
  static String get allToolsLabel => 'All tools ($toolCount)';

  static const _items = <_FeatureItem>[
    _FeatureItem(
      'Translate contract',
      'OCR + automatic MT (Google → LFM)',
      Icons.translate,
      TranslateContractScreen(),
    ),
    _FeatureItem('E-sign (in-app record)', 'Record consent + signer name', Icons.draw_outlined, EsignFeatureScreen()),
    _FeatureItem('Compare A vs B', 'Diff files or two saved analyses', Icons.compare_arrows, CompareFeatureScreen()),
    _FeatureItem('AI Assistant', 'General Q&A or negotiation coach (Gemini)', Icons.chat_outlined, ChatAssistantScreen()),
    _FeatureItem('Explain clause', 'Paste clause text + optional analysis id', Icons.menu_book_outlined, ExplainClauseFeatureScreen()),
    _FeatureItem('Risk dashboard', 'Scores from saved analysis', Icons.analytics_outlined, RiskFeatureScreen()),
    _FeatureItem('Summarize clauses', 'Batch summaries via LFM', Icons.summarize_outlined, SummarizeFeatureScreen()),
    _FeatureItem('Share analysis', 'Read-only share link', Icons.share_outlined, ShareFeatureScreen()),
    _FeatureItem('Contract timeline', 'Milestones (admin)', Icons.timeline_outlined, TimelineAdminFeatureScreen()),
    _FeatureItem('Biometrics', 'Face ID info — JWT storage notes', Icons.fingerprint_outlined, BiometricInfoScreen()),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: LegatoAppBar(
        title: Text('Tools (${_items.length})'),
      ),
      body: GridView.builder(
        padding: const EdgeInsets.all(12),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          mainAxisExtent: 132,
          crossAxisSpacing: 10,
          mainAxisSpacing: 10,
        ),
        itemCount: _items.length,
        itemBuilder: (context, i) {
          final it = _items[i];

          return Material(
            color: Theme.of(context).colorScheme.surface,
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
              side: BorderSide(
                color: Theme.of(context).brightness == Brightness.dark
                    ? const Color(0xFF30363D)
                    : LegatoLinkedInTheme.border,
              ),
            ),
            child: InkWell(
              onTap: () => Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => it.screen),
              ),
              borderRadius: BorderRadius.circular(8),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(10, 12, 10, 10),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      it.icon,
                      size: 26,
                      color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      it.title,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w600,
                            height: 1.2,
                          ),
                    ),
                    const Spacer(),
                    Text(
                      it.subtitle,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                            fontSize: 11,
                            height: 1.25,
                          ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _FeatureItem {
  const _FeatureItem(this.title, this.subtitle, this.icon, this.screen);
  final String title;
  final String subtitle;
  final IconData icon;
  final Widget screen;
}
