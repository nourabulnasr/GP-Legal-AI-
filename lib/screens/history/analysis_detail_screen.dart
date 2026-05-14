import 'dart:convert';

import 'package:flutter/material.dart';

import 'package:legato_mobile/screens/chat/chat_analysis_screen.dart';
import 'package:legato_mobile/screens/features/phase5_screens.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class AnalysisDetailScreen extends StatelessWidget {
  const AnalysisDetailScreen({
    super.key,
    required this.title,
    required this.payload,
    this.analysisId,
  });

  final String title;
  final Map<String, dynamic> payload;
  final int? analysisId;

  @override
  Widget build(BuildContext context) {
    final hits = payload['rule_hits'];
    final hitsAsList = hits is List ? hits : <dynamic>[];
    final needsReview = payload['needs_review'] == true;
    final labor = payload['labor_summary'];
    final unifiedRisk = payload['full_text_unified_risk'];
    final ragByViolation = payload['rag_by_violation'];
    final ragList = ragByViolation is List ? ragByViolation : <dynamic>[];

    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: Text(title, maxLines: 1, overflow: TextOverflow.ellipsis),
          actions: [
            if (analysisId != null)
              IconButton(
                tooltip: 'Chat about this analysis',
                icon: const Icon(Icons.chat_bubble_outline),
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => ChatAnalysisScreen(initialAnalysisId: analysisId),
                  ),
                ),
              ),
          ],
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Violations'),
              Tab(text: 'Summary'),
              Tab(text: 'RAG hits'),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            _ViolationsTab(
              hits: hitsAsList,
              needsReview: needsReview,
              unifiedRisk: unifiedRisk,
              analysisId: analysisId,
            ),
            _SummaryTab(labor: labor, payload: payload),
            _RagTab(ragList: ragList),
          ],
        ),
      ),
    );
  }
}

// ── Violations tab ────────────────────────────────────────────────────────────

class _ViolationsTab extends StatelessWidget {
  const _ViolationsTab({
    required this.hits,
    required this.needsReview,
    this.unifiedRisk,
    this.analysisId,
  });

  final List<dynamic> hits;
  final bool needsReview;
  final dynamic unifiedRisk;
  final int? analysisId;

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        if (needsReview)
          Container(
            margin: const EdgeInsets.only(bottom: 16),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: const Color(0xFFFEF3C7),
              border: Border.all(color: const Color(0xFFF59E0B)),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Row(
              children: [
                Icon(Icons.warning_amber_rounded, color: Color(0xFFB45309), size: 20),
                SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Lawyer review recommended — one or more high-severity violations detected.',
                    style: TextStyle(color: Color(0xFF92400E), fontSize: 13, fontWeight: FontWeight.w500),
                  ),
                ),
              ],
            ),
          ),
        if (unifiedRisk != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Row(
              children: [
                const Text('ML risk score: ', style: TextStyle(fontWeight: FontWeight.w500)),
                Text(unifiedRisk.toString(), style: TextStyle(color: LegatoLinkedInTheme.textSecondaryAdaptive(context))),
              ],
            ),
          ),
        if (hits.isEmpty)
          Center(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                children: [
                  const Icon(Icons.check_circle_outline, size: 48, color: Color(0xFF059669)),
                  const SizedBox(height: 12),
                  const Text('No violations found', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  Text('The contract passed all rule checks.', style: TextStyle(color: LegatoLinkedInTheme.textSecondaryAdaptive(context))),
                ],
              ),
            ),
          )
        else ...[
          Text(
            '${hits.length} violation${hits.length == 1 ? '' : 's'} found',
            style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
          ),
          const SizedBox(height: 10),
          for (final raw in hits)
            if (raw is Map)
              _ViolationCard(hit: Map<String, dynamic>.from(raw), analysisId: analysisId),
        ],
      ],
    );
  }
}

class _ViolationCard extends StatelessWidget {
  const _ViolationCard({required this.hit, this.analysisId});

  final Map<String, dynamic> hit;
  final int? analysisId;

  Color _severityColor(BuildContext context, String? sev) {
    switch (sev?.toLowerCase()) {
      case 'error':
      case 'critical':
        return const Color(0xFFB24020);
      case 'warning':
      case 'high':
        return const Color(0xFFD97706);
      case 'info':
      case 'low':
        return const Color(0xFF0369A1);
      default:
        return LegatoLinkedInTheme.textSecondaryAdaptive(context);
    }
  }

  Color _severityBg(String? sev) {
    switch (sev?.toLowerCase()) {
      case 'error':
      case 'critical':
        return const Color(0xFFFEF2F2);
      case 'warning':
      case 'high':
        return const Color(0xFFFFFBEB);
      case 'info':
      case 'low':
        return const Color(0xFFEFF6FF);
      default:
        return Colors.grey.shade50;
    }
  }

  String _label(String? sev) {
    if (sev == null) return 'Rule hit';
    return sev[0].toUpperCase() + sev.substring(1).toLowerCase();
  }

  String _title() {
    for (final k in ['rule_id', 'article', 'id', 'heading']) {
      final v = hit[k];
      if (v != null && '$v'.trim().isNotEmpty) return '$v';
    }
    return 'Violation';
  }

  String _description() {
    for (final k in ['summary', 'description', 'rationale', 'text', 'snippet']) {
      final v = hit[k];
      if (v != null && '$v'.trim().isNotEmpty) return '$v';
    }
    return '';
  }

  String _clauseText() {
    for (final k in ['clause_text', 'text', 'snippet', 'summary']) {
      final v = hit[k];
      if (v != null && '$v'.trim().isNotEmpty) return '$v';
    }
    return _description();
  }

  @override
  Widget build(BuildContext context) {
    final sev = hit['severity']?.toString();
    final color = _severityColor(context, sev);
    final bg = _severityBg(sev);
    final desc = _description();
    final explanation = hit['explanation']?.toString() ?? '';

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: bg,
        border: Border(left: BorderSide(color: color, width: 4)),
        borderRadius: const BorderRadius.only(
          topRight: Radius.circular(8),
          bottomRight: Radius.circular(8),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Text(
                    _title(),
                    style: TextStyle(fontWeight: FontWeight.w700, color: color, fontSize: 13),
                  ),
                ),
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    _label(sev),
                    style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
            if (desc.isNotEmpty) ...[
              const SizedBox(height: 6),
              Text(desc, style: const TextStyle(fontSize: 13, height: 1.4)),
            ],
            if (explanation.isNotEmpty) ...[
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surface,
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                              Text('LFM explanation', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: LegatoLinkedInTheme.textSecondaryAdaptive(context))),
                    const SizedBox(height: 4),
                    Text(explanation, style: const TextStyle(fontSize: 13, height: 1.4)),
                  ],
                ),
              ),
            ],
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(
                style: TextButton.styleFrom(
                  foregroundColor: color,
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
                onPressed: () {
                  final text = _clauseText();
                  Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => ExplainClauseFeatureScreen(
                        initialClauseText: text,
                        initialAnalysisId: analysisId,
                      ),
                    ),
                  );
                },
                icon: const Icon(Icons.menu_book_outlined, size: 16),
                label: const Text('Explain with LFM', style: TextStyle(fontSize: 12)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Summary tab ───────────────────────────────────────────────────────────────

class _SummaryTab extends StatelessWidget {
  const _SummaryTab({required this.labor, required this.payload});

  final dynamic labor;
  final Map<String, dynamic> payload;

  @override
  Widget build(BuildContext context) {
    final summary = payload['summary']?.toString() ?? payload['labor_summary_text']?.toString() ?? '';

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        if (summary.isNotEmpty) ...[
          Text('Summary', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          SelectableText(summary),
          const SizedBox(height: 20),
        ],
        if (labor != null)
          ExpansionTile(
            title: const Text('Labor law analysis'),
            children: [
              Padding(
                padding: const EdgeInsets.all(12),
                child: SelectableText(const JsonEncoder.withIndent('  ').convert(labor)),
              ),
            ],
          ),
      ],
    );
  }
}

// ── RAG hits tab ──────────────────────────────────────────────────────────────

class _RagTab extends StatelessWidget {
  const _RagTab({required this.ragList});

  final List<dynamic> ragList;

  @override
  Widget build(BuildContext context) {
    if (ragList.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Text('No RAG hits for this analysis.', style: TextStyle(color: LegatoLinkedInTheme.textSecondaryAdaptive(context))),
        ),
      );
    }
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: ragList.length,
      itemBuilder: (context, i) {
        final block = ragList[i];
        if (block is! Map) return const SizedBox.shrink();
        final b = Map<String, dynamic>.from(block);
        final ruleId = b['rule_id']?.toString() ?? 'Rule ${i + 1}';
        final hits = b['hits'];
        final hitList = hits is List ? hits : <dynamic>[];

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (i > 0) const SizedBox(height: 16),
            Row(
              children: [
                const Icon(Icons.gavel_outlined, size: 15, color: LegatoLinkedInTheme.navActiveGold),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    ruleId,
                    style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13),
                  ),
                ),
                Text(
                  '${hitList.length} article${hitList.length == 1 ? '' : 's'}',
                  style: TextStyle(fontSize: 11, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                ),
              ],
            ),
            const SizedBox(height: 8),
            for (final raw in hitList)
              if (raw is Map)
                _RagHitTile(hit: Map<String, dynamic>.from(raw)),
            if (i < ragList.length - 1) const Divider(height: 24),
          ],
        );
      },
    );
  }
}

class _RagHitTile extends StatelessWidget {
  const _RagHitTile({required this.hit});

  final Map<String, dynamic> hit;

  @override
  Widget build(BuildContext context) {
    final meta = hit['metadata'];
    final metaMap = meta is Map ? Map<String, dynamic>.from(meta) : <String, dynamic>{};
    final article = hit['article']?.toString() ?? metaMap['article']?.toString() ?? metaMap['source']?.toString() ?? '';
    final text = hit['text']?.toString() ?? hit['content']?.toString() ?? '';
    final score = hit['score']?.toString() ?? metaMap['score']?.toString() ?? '';

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (article.isNotEmpty)
            Row(
              children: [
                Text(
                  'Article $article',
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                    color: LegatoLinkedInTheme.navActiveGold,
                  ),
                ),
                if (score.isNotEmpty) ...[
                  const Spacer(),
                  Text(
                    score,
                    style: TextStyle(fontSize: 11, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                  ),
                ],
              ],
            ),
          if (article.isNotEmpty && text.isNotEmpty) const SizedBox(height: 4),
          if (text.isNotEmpty)
            Text(
              text,
              style: TextStyle(fontSize: 12, height: 1.4, color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
            ),
        ],
      ),
    );
  }
}
