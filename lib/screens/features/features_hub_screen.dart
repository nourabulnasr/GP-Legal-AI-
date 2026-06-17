import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/chat/chat_assistant_screen.dart';
import 'package:legato_mobile/screens/features/phase5_screens.dart';
import 'package:legato_mobile/screens/lawyer/lawyer_application_screen.dart';
import 'package:legato_mobile/screens/translate/translate_contract_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

/// Entry to all Phase-5 capabilities (wired to `/legato/*` API).
class FeaturesHubScreen extends StatelessWidget {
  const FeaturesHubScreen({super.key});

  // Count of always-visible tools (excluding the conditional lawyer item).
  // Used by external screens for labels; off-by-one is acceptable when the
  // lawyer tool is also shown.
  static int get toolCount => _baseItems.length;
  static String get openAllToolsLabel => 'Open all tools ($toolCount)';
  static String get allToolsLabel => 'All tools ($toolCount)';

  static const _baseItems = <_FeatureItem>[
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

  static const _lawyerItem = _FeatureItem(
    'Lawyer Verification',
    'Apply or check your verification status',
    Icons.gavel_outlined,
    LawyerApplicationScreen(),
  );

  @override
  Widget build(BuildContext context) {
    final user = context.watch<AuthProvider>().user;

    // Admins manage applications — they don't apply.
    // Verified lawyers can still tap it to view their approved status.
    final items = (user != null && !user.isAdmin)
        ? [..._baseItems, _lawyerItem]
        : _baseItems;

    return Scaffold(
      appBar: LegatoAppBar(
        title: Text('Tools (${items.length})'),
      ),
      body: GridView.builder(
        padding: const EdgeInsets.all(12),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          mainAxisExtent: 132,
          crossAxisSpacing: 10,
          mainAxisSpacing: 10,
        ),
        itemCount: items.length,
        itemBuilder: (context, i) {
          final it = items[i];
          final isLawyerTool = it == _lawyerItem;
          final isVerified = user?.isVerifiedLawyer ?? false;

          return Material(
            color: Theme.of(context).colorScheme.surface,
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
              side: BorderSide(
                color: isLawyerTool
                    ? (isVerified
                        ? Colors.green.withValues(alpha: 0.45)
                        : LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.6))
                    : (Theme.of(context).brightness == Brightness.dark
                        ? const Color(0xFF30363D)
                        : LegatoLinkedInTheme.border),
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
                    Row(
                      children: [
                        Icon(
                          it.icon,
                          size: 26,
                          color: isLawyerTool
                              ? (isVerified ? Colors.green : LegatoLinkedInTheme.navActiveGold)
                              : LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95),
                        ),
                        if (isLawyerTool && isVerified) ...[
                          const SizedBox(width: 4),
                          const Icon(Icons.verified, size: 14, color: Colors.green),
                        ],
                      ],
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
                      isLawyerTool && isVerified ? 'Verified lawyer — tap to view status' : it.subtitle,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: isLawyerTool && isVerified
                                ? Colors.green.withValues(alpha: 0.85)
                                : LegatoLinkedInTheme.textSecondaryAdaptive(context),
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
