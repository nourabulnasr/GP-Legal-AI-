import 'package:flutter/material.dart';

/// Phase 5 product features from the Legato roadmap (backend work varies per item).
class RoadmapScreen extends StatelessWidget {
  const RoadmapScreen({super.key});

  static const _items = <_RoadmapItem>[
    _RoadmapItem(
      'E-sign',
      'Sign contracts in-app; usually needs a provider API + audit storage.',
      'Tier C — backend TBD',
    ),
    _RoadmapItem(
      'Contract comparison (A vs B)',
      'Diff two versions; needs compare endpoint or two analysis IDs.',
      'Tier B — backend TBD',
    ),
    _RoadmapItem(
      'Voice assistant',
      'Speech-to-text on device, then existing chat/analyze APIs.',
      'Tier D — use mic + /chat/*',
    ),
    _RoadmapItem(
      'Tap clause → legal explanation',
      'Reader UI + /explain/clause (or extend chat with clause context).',
      'Tier A/B — backend TBD',
    ),
    _RoadmapItem(
      'Risk scoring + smart notifications',
      'Dashboard score + FCM; needs prefs + worker for pushes.',
      'Tier B — partial: show risks from analysis JSON today',
    ),
    _RoadmapItem(
      'Face ID / biometrics',
      'Use `local_auth` to protect app / step-up (mostly client-side).',
      'Tier D — Settings: test biometric',
    ),
    _RoadmapItem(
      'Clause summarization',
      'Batch summarize clauses; needs /summarize/clauses or LLM prompt mode.',
      'Tier A — backend TBD',
    ),
    _RoadmapItem(
      'AI negotiation assistant',
      'Negotiation chat mode + disclaimers; extend chat router.',
      'Tier B — backend TBD',
    ),
    _RoadmapItem(
      'Share analysis',
      'Share links / invites; needs shares table + auth on GET.',
      'Tier B — backend TBD',
    ),
    _RoadmapItem(
      'Contract timeline (admin)',
      'Milestones from text or manual tags; extend admin APIs.',
      'Tier B/E — backend TBD',
    ),
    _RoadmapItem(
      'Chat with other party',
      'Threads per contract; WebSocket or poll + moderation policy.',
      'Tier B/E — backend TBD',
    ),
    _RoadmapItem(
      'Legal LinkedIn-style network',
      'Profiles, orgs, feed — separate product pillar; Postgres-scale.',
      'Tier E — not started',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Roadmap')),
      body: ListView.separated(
        padding: const EdgeInsets.all(16),
        itemCount: _items.length,
        separatorBuilder: (context, i) => const SizedBox(height: 8),
        itemBuilder: (context, i) {
          final it = _items[i];
          return Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('${i + 1}. ${it.title}', style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 6),
                  Text(it.body, style: Theme.of(context).textTheme.bodyMedium),
                  const SizedBox(height: 6),
                  Text(
                    it.status,
                    style: Theme.of(context).textTheme.labelSmall?.copyWith(
                          color: Theme.of(context).colorScheme.primary,
                        ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}

class _RoadmapItem {
  const _RoadmapItem(this.title, this.body, this.status);
  final String title;
  final String body;
  final String status;
}
