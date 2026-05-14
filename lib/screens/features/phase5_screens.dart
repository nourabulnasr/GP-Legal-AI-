import 'dart:convert' show JsonEncoder, base64Encode, utf8;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hand_signature/signature.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/config/app_config.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

// --- 1 E-sign ---

class EsignFeatureScreen extends StatefulWidget {
  const EsignFeatureScreen({super.key});

  @override
  State<EsignFeatureScreen> createState() => _EsignFeatureScreenState();
}

class _EsignFeatureScreenState extends State<EsignFeatureScreen> {
  final _analysisId = TextEditingController();
  final _name = TextEditingController();
  final _sigCtrl = HandSignatureControl(
    initialSetup: SignaturePathSetup(
      threshold: 3.0,
      smoothRatio: 0.65,
      velocityRange: 2.0,
    ),
  );
  bool _consent = false;
  bool _busy = false;
  String? _err;
  String? _ok;

  @override
  void dispose() {
    _analysisId.dispose();
    _name.dispose();
    _sigCtrl.dispose();
    super.dispose();
  }

  Future<String?> _canvasToBase64() async {
    // hand_signature 3.x: toImage() returns Future<ByteData?> (PNG-encoded directly)
    final bytes = await _sigCtrl.toImage(width: 600, height: 200);
    if (bytes == null) return null;
    return base64Encode(bytes.buffer.asUint8List());
  }

  Future<void> _submit() async {
    final id = int.tryParse(_analysisId.text.trim());
    if (id == null || _name.text.trim().isEmpty) {
      setState(() => _err = 'Analysis id and signer name required.');
      return;
    }
    if (!_consent) {
      setState(() => _err = 'Acknowledge consent to continue.');
      return;
    }
    if (!_sigCtrl.hasActivePath) {
      setState(() => _err = 'Please draw your signature in the canvas.');
      return;
    }
    setState(() {
      _busy = true;
      _err = null;
      _ok = null;
    });
    try {
      final b64 = await _canvasToBase64();
      if (!mounted) return;
      final r = await context.read<AppServices>().legato.recordSignature(
            analysisId: id,
            signerName: _name.text.trim(),
            consentAcknowledged: true,
            signaturePngBase64: b64,
          );
      final rid = r['id'];
      final created = r['created_at'];
      setState(() {
        _ok = rid != null
            ? 'Signature record #$rid saved.${created != null ? ' · $created' : ''}'
            : 'Saved.';
      });
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('E-sign (in-app record)')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            'Stores an acknowledgment record in the Legato database. Draw your signature below.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _analysisId,
            decoration: const InputDecoration(labelText: 'Analysis id', border: OutlineInputBorder()),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _name,
            decoration: const InputDecoration(labelText: 'Signer name', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Signature', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
              TextButton.icon(
                onPressed: () => _sigCtrl.clear(),
                icon: const Icon(Icons.refresh, size: 16),
                label: const Text('Clear'),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Container(
            height: 160,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0xFFD1D5DB), width: 1.5),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: HandSignature(
                control: _sigCtrl,
                drawer: ShapeSignatureDrawer(
                  color: const Color(0xFF1B1F23),
                  width: 2.0,
                  maxWidth: 6.0,
                ),
              ),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Draw your signature in the box above',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary),
          ),
          const SizedBox(height: 12),
          CheckboxListTile(
            value: _consent,
            onChanged: (v) => setState(() => _consent = v ?? false),
            title: const Text('I acknowledge this is an in-app record, not a certified e-signature.'),
            contentPadding: EdgeInsets.zero,
          ),
          const SizedBox(height: 8),
          FilledButton(
            onPressed: _busy ? null : _submit,
            child: _busy
                ? const SizedBox(height: 22, width: 22, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Text('Submit record'),
          ),
          if (_err != null) ...[
            const SizedBox(height: 8),
            Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ],
          if (_ok != null) ...[
            const SizedBox(height: 12),
            Text(_ok!, style: Theme.of(context).textTheme.titleMedium?.copyWith(color: const Color(0xFF059669))),
          ],
        ],
      ),
    );
  }
}

// --- 2 Compare ---

class CompareFeatureScreen extends StatefulWidget {
  const CompareFeatureScreen({super.key});

  @override
  State<CompareFeatureScreen> createState() => _CompareFeatureScreenState();
}

class _CompareFeatureScreenState extends State<CompareFeatureScreen> {
  final _a = TextEditingController();
  final _b = TextEditingController();
  final _idA = TextEditingController();
  final _idB = TextEditingController();
  bool _busy = false;
  String? _err;
  String? _comparison;

  @override
  void dispose() {
    _a.dispose();
    _b.dispose();
    _idA.dispose();
    _idB.dispose();
    super.dispose();
  }

  Future<void> _pick(bool left) async {
    final r = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['txt'],
      withData: true,
    );
    if (r == null || r.files.isEmpty) return;
    final f = r.files.single;
    final bytes = f.bytes;
    if (bytes == null) return;
    if (bytes.length > 300000) {
      setState(() => _err = 'File too large. Paste the contract text directly or use an analysis ID.');
      return;
    }
    final txt = utf8.decode(bytes, allowMalformed: false);
    setState(() {
      _err = null;
      if (left) {
        _a.text = txt;
      } else {
        _b.text = txt;
      }
    });
  }

  Future<void> _run() async {
    setState(() {
      _busy = true;
      _err = null;
      _comparison = null;
    });
    try {
      final api = context.read<AppServices>().legato;
      final ia = int.tryParse(_idA.text.trim());
      final ib = int.tryParse(_idB.text.trim());
      final res = await api.compareContracts(
        textA: _a.text.isEmpty ? null : _a.text,
        textB: _b.text.isEmpty ? null : _b.text,
        analysisIdA: ia,
        analysisIdB: ib,
      );
      setState(() {
        _comparison = res['comparison']?.toString();
      });
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: LegatoLinkedInTheme.background,
      appBar: AppBar(title: const Text('Compare contracts')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            'Load a plain .txt file, or paste contract text directly. For PDF/DOCX contracts, use two analysis IDs instead.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(child: OutlinedButton(onPressed: () => _pick(true), child: const Text('Load A'))),
              const SizedBox(width: 8),
              Expanded(child: OutlinedButton(onPressed: () => _pick(false), child: const Text('Load B'))),
            ],
          ),
          TextField(controller: _a, decoration: const InputDecoration(labelText: 'Text A (or leave empty if using id A)'), maxLines: 4),
          TextField(controller: _b, decoration: const InputDecoration(labelText: 'Text B (or leave empty if using id B)'), maxLines: 4),
          TextField(controller: _idA, decoration: const InputDecoration(labelText: 'Optional analysis id A'), keyboardType: TextInputType.number),
          TextField(controller: _idB, decoration: const InputDecoration(labelText: 'Optional analysis id B'), keyboardType: TextInputType.number),
          const SizedBox(height: 8),
          FilledButton(onPressed: _busy ? null : _run, child: _busy ? const SizedBox(height: 22, width: 22, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Compare')),
          if (_err != null) ...[
            const SizedBox(height: 8),
            Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ],
          if (_comparison != null && _comparison!.trim().isNotEmpty) ...[
            const SizedBox(height: 16),
            Text(
              'Comparison',
              style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(14),
                child: SelectableText(
                  _comparison!,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(height: 1.5),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// --- 3 Voice (text transcript; no speech_to_text — avoids native hooks when Pub cache path has spaces) ---

class VoiceAssistantFeatureScreen extends StatefulWidget {
  const VoiceAssistantFeatureScreen({super.key});

  @override
  State<VoiceAssistantFeatureScreen> createState() => _VoiceAssistantFeatureScreenState();
}

class _VoiceAssistantFeatureScreenState extends State<VoiceAssistantFeatureScreen> {
  final _text = TextEditingController();
  String? _err;
  final List<Map<String, dynamic>> _hist = [];

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  Future<void> _sendToAssistant() async {
    if (_text.text.trim().isEmpty) return;
    setState(() => _err = null);
    try {
      final r = await context.read<AppServices>().legato.chatAssistant(
            message: _text.text.trim(),
            history: _hist,
          );
      final reply = r['content']?.toString() ?? '';
      setState(() {
        _hist.add({'role': 'user', 'content': _text.text.trim()});
        _hist.add({'role': 'assistant', 'content': reply});
        _text.clear();
      });
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Assistant (transcript)')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Type what you would say (speech-to-text was removed so builds work when your Windows user folder has a space). '
            'Sends to POST /chat/assistant.',
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _text,
            decoration: const InputDecoration(
              labelText: 'Message',
              hintText: 'Your question…',
            ),
            maxLines: 3,
          ),
          const SizedBox(height: 8),
          FilledButton(onPressed: _sendToAssistant, child: const Text('Send to assistant')),
          if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          for (final m in _hist) ListTile(title: Text(m['role'] ?? ''), subtitle: Text(m['content'] ?? '')),
        ],
      ),
    );
  }
}

// --- 4 Explain clause ---

class ExplainClauseFeatureScreen extends StatefulWidget {
  const ExplainClauseFeatureScreen({
    super.key,
    this.initialClauseText,
    this.initialAnalysisId,
  });

  final String? initialClauseText;
  final int? initialAnalysisId;

  @override
  State<ExplainClauseFeatureScreen> createState() => _ExplainClauseFeatureScreenState();
}

class _ExplainClauseFeatureScreenState extends State<ExplainClauseFeatureScreen> {
  final _clause = TextEditingController();
  final _aid = TextEditingController();
  String _language = 'auto'; // auto | ar | en
  bool _busy = false;
  String? _out;
  String? _err;

  @override
  void initState() {
    super.initState();
    if (widget.initialClauseText != null) _clause.text = widget.initialClauseText!;
    if (widget.initialAnalysisId != null) _aid.text = widget.initialAnalysisId.toString();
  }

  @override
  void dispose() {
    _clause.dispose();
    _aid.dispose();
    super.dispose();
  }

  Future<void> _go() async {
    setState(() {
      _busy = true;
      _err = null;
      _out = null;
    });
    try {
      final id = int.tryParse(_aid.text.trim());
      final r = await context.read<AppServices>().legato.explainClause(
            clauseText: _clause.text,
            analysisId: id,
            language: _language == 'auto' ? null : _language,
          );
      if (!mounted) return;
      setState(() => _out = r['explanation']?.toString());
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _err = e.message);
    } catch (e) {
      if (!mounted) return;
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: LegatoLinkedInTheme.background,
      appBar: AppBar(title: const Text('Explain clause')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            'Paste the clause text. Add an analysis id for richer context (OCR + rule hits).',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary),
          ),
          const SizedBox(height: 12),
          TextField(controller: _clause, decoration: const InputDecoration(labelText: 'Clause text'), maxLines: 8),
          TextField(controller: _aid, decoration: const InputDecoration(labelText: 'Optional analysis id'), keyboardType: TextInputType.number),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            initialValue: _language,
            decoration: const InputDecoration(labelText: 'Explanation language'),
            items: const [
              DropdownMenuItem(value: 'auto', child: Text('Auto (detect)')),
              DropdownMenuItem(value: 'ar', child: Text('Arabic')),
              DropdownMenuItem(value: 'en', child: Text('English')),
            ],
            onChanged: _busy
                ? null
                : (v) {
                    if (v != null) setState(() => _language = v);
                  },
          ),
          const SizedBox(height: 8),
          FilledButton(
            onPressed: _busy ? null : _go,
            child: _busy
                ? const SizedBox(height: 22, width: 22, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Text('Explain'),
          ),
          if (_err != null) ...[
            const SizedBox(height: 8),
            Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ],
          if (_out != null) ...[
            const SizedBox(height: 16),
            Text('Explanation', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(14),
                child: SelectableText(_out!, style: Theme.of(context).textTheme.bodyMedium?.copyWith(height: 1.45)),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Informational only — not legal advice.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondary),
            ),
          ],
        ],
      ),
    );
  }
}

// --- 5 Risk dashboard ---

class RiskFeatureScreen extends StatefulWidget {
  const RiskFeatureScreen({super.key});

  @override
  State<RiskFeatureScreen> createState() => _RiskFeatureScreenState();
}

class _RiskFeatureScreenState extends State<RiskFeatureScreen> {
  final _id = TextEditingController();
  bool _busy = false;
  Map<String, dynamic>? _data;
  String? _err;

  @override
  void dispose() {
    _id.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    final i = int.tryParse(_id.text.trim());
    if (i == null) return;
    setState(() {
      _busy = true;
      _err = null;
      _data = null;
    });
    try {
      final r = await context.read<AppServices>().legato.riskSummary(i);
      setState(() => _data = r);
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Risk summary')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _id,
            decoration: const InputDecoration(
              labelText: 'Analysis id',
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.number,
          ),
          const SizedBox(height: 8),
          FilledButton(
            onPressed: _busy ? null : _load,
            child: _busy
                ? const SizedBox(height: 22, width: 22, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Text('Load risk summary'),
          ),
          if (_err != null) ...[
            const SizedBox(height: 8),
            Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ],
          if (_data != null) _RiskResultCard(data: _data!),
        ],
      ),
    );
  }
}

class _RiskResultCard extends StatelessWidget {
  const _RiskResultCard({required this.data});
  final Map<String, dynamic> data;

  Color _levelColor(String level) {
    switch (level) {
      case 'critical':
        return const Color(0xFFB24020);
      case 'high':
        return const Color(0xFFD97706);
      case 'medium':
        return const Color(0xFF2563EB);
      default:
        return const Color(0xFF059669);
    }
  }

  String _riskLevel(Map<String, dynamic> d) {
    final errors = (d['error_count'] as num?)?.toInt() ?? 0;
    final warnings = (d['warning_count'] as num?)?.toInt() ?? 0;
    final mlRisk = (d['unified_ml_risk'] as num?)?.toDouble() ?? 0.0;
    if (errors > 0 || mlRisk >= 0.7) return 'critical';
    if (warnings > 1 || mlRisk >= 0.4) return 'high';
    if (warnings > 0 || mlRisk >= 0.2) return 'medium';
    return 'low';
  }

  @override
  Widget build(BuildContext context) {
    final errors = (data['error_count'] as num?)?.toInt() ?? 0;
    final warnings = (data['warning_count'] as num?)?.toInt() ?? 0;
    final info = (data['info_count'] as num?)?.toInt() ?? 0;
    final total = (data['total_hits'] as num?)?.toInt() ?? (errors + warnings + info);
    final mlRisk = (data['unified_ml_risk'] as num?)?.toDouble();
    final hits = (data['rule_hits'] as List<dynamic>?) ?? [];
    final needsReview = data['needs_review'] == true;
    final level = _riskLevel(data);
    final color = _levelColor(level);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 16),
        // Overall badge
        Container(
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withValues(alpha: 0.4), width: 1.5),
          ),
          child: Row(
            children: [
              Icon(Icons.shield_outlined, color: color, size: 32),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Risk level: ${level.toUpperCase()}',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: color),
                    ),
                    if (mlRisk != null)
                      Text(
                        'ML score: ${(mlRisk * 100).toStringAsFixed(0)}%',
                        style: TextStyle(color: color.withValues(alpha: 0.85), fontSize: 13),
                      ),
                  ],
                ),
              ),
              if (needsReview)
                Chip(
                  label: const Text('Needs Review', style: TextStyle(fontSize: 11)),
                  backgroundColor: const Color(0xFFB24020).withValues(alpha: 0.12),
                ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        // Stats row
        Row(
          children: [
            _RiskStat(label: 'Errors', value: errors, color: const Color(0xFFB24020)),
            const SizedBox(width: 8),
            _RiskStat(label: 'Warnings', value: warnings, color: const Color(0xFFD97706)),
            const SizedBox(width: 8),
            _RiskStat(label: 'Total hits', value: total, color: LegatoLinkedInTheme.textSecondary),
          ],
        ),
        if (hits.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text(
            'Top violations',
            style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          ...hits.take(3).map((h) {
            final hit = h is Map ? Map<String, dynamic>.from(h) : <String, dynamic>{};
            final sev = hit['severity']?.toString() ?? 'info';
            final sevColor = sev == 'error' ? const Color(0xFFB24020) : sev == 'warning' ? const Color(0xFFD97706) : const Color(0xFF2563EB);
            return Card(
              margin: const EdgeInsets.only(bottom: 8),
              child: ListTile(
                leading: Container(
                  width: 8,
                  height: 40,
                  decoration: BoxDecoration(color: sevColor, borderRadius: BorderRadius.circular(4)),
                ),
                title: Text(
                  hit['rule_id']?.toString() ?? hit['description']?.toString() ?? 'Violation',
                  style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
                ),
                subtitle: hit['description'] != null && hit['rule_id'] != null
                    ? Text(hit['description'].toString(), maxLines: 2, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 12))
                    : null,
                trailing: Chip(
                  label: Text(sev, style: const TextStyle(fontSize: 11)),
                  backgroundColor: sevColor.withValues(alpha: 0.12),
                  padding: EdgeInsets.zero,
                ),
              ),
            );
          }),
        ],
      ],
    );
  }
}

class _RiskStat extends StatelessWidget {
  const _RiskStat({required this.label, required this.value, required this.color});
  final String label;
  final int value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.07),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: color.withValues(alpha: 0.25)),
        ),
        child: Column(
          children: [
            Text('$value', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: color)),
            Text(label, style: TextStyle(fontSize: 11, color: LegatoLinkedInTheme.textSecondary)),
          ],
        ),
      ),
    );
  }
}

// --- 6 Biometrics (info only; local_auth removed — it pulled objective_c native hooks that break if Pub path has spaces) ---

class BiometricInfoScreen extends StatelessWidget {
  const BiometricInfoScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Biometrics')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            'Face ID / fingerprint',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          const Text(
            'The local_auth package was removed from this project so Android builds succeed when your '
            'Windows username contains a space (e.g. C:\\Users\\Aly ahmed\\…). That dependency pulled '
            'native asset hooks that failed with "C:\\Users\\Aly is not recognized".\n\n'
            'To add biometrics later: move PUB_CACHE to a path without spaces (e.g. C:\\dev\\pub-cache), '
            'then add local_auth again.',
          ),
          const SizedBox(height: 16),
          Text(
            'JWT storage',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: 4),
          const Text(
            'Tokens still use SharedPreferences. For production, use flutter_secure_storage after fixing Pub cache path.',
          ),
        ],
      ),
    );
  }
}

// --- 7 Summarize ---

class SummarizeFeatureScreen extends StatefulWidget {
  const SummarizeFeatureScreen({super.key});

  @override
  State<SummarizeFeatureScreen> createState() => _SummarizeFeatureScreenState();
}

class _SummarizeFeatureScreenState extends State<SummarizeFeatureScreen> {
  final _text = TextEditingController();
  final _aid = TextEditingController();
  bool _busy = false;
  String? _out;
  String? _err;

  @override
  void dispose() {
    _text.dispose();
    _aid.dispose();
    super.dispose();
  }

  Future<void> _go() async {
    final parts = _text.text.split('\n---\n');
    if (parts.isEmpty || parts.every((e) => e.trim().isEmpty)) {
      setState(() => _err = 'Enter clauses separated by ---');
      return;
    }
    setState(() {
      _busy = true;
      _err = null;
      _out = null;
    });
    try {
      final id = int.tryParse(_aid.text.trim());
      final r = await context.read<AppServices>().legato.summarizeClauses(
            clauses: parts.map((e) => e.trim()).where((e) => e.isNotEmpty).toList(),
            analysisId: id,
          );
      final sums = r['summaries'];
      setState(() => _out = const JsonEncoder.withIndent('  ').convert(sums));
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Summarize clauses')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text('Separate clauses with a line containing only ---'),
          TextField(controller: _text, maxLines: 10, decoration: const InputDecoration(labelText: 'Clauses')),
          TextField(controller: _aid, decoration: const InputDecoration(labelText: 'Optional analysis id'), keyboardType: TextInputType.number),
          FilledButton(onPressed: _busy ? null : _go, child: const Text('Summarize')),
          if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          if (_out != null) SelectableText(_out!),
        ],
      ),
    );
  }
}

// --- 8 Negotiation ---

class NegotiationFeatureScreen extends StatefulWidget {
  const NegotiationFeatureScreen({super.key});

  @override
  State<NegotiationFeatureScreen> createState() => _NegotiationFeatureScreenState();
}

class _NegotiationFeatureScreenState extends State<NegotiationFeatureScreen> {
  final _msg = TextEditingController();
  final _aid = TextEditingController();
  final List<Map<String, dynamic>> _hist = [];
  bool _busy = false;
  String? _err;

  @override
  void dispose() {
    _msg.dispose();
    _aid.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    if (_msg.text.trim().isEmpty) return;
    setState(() {
      _busy = true;
      _err = null;
    });
    try {
      final id = int.tryParse(_aid.text.trim());
      final r = await context.read<AppServices>().legato.negotiationChat(
            message: _msg.text,
            analysisId: id,
            history: _hist,
          );
      final c = r['content']?.toString() ?? '';
      setState(() {
        _hist.add({'role': 'user', 'content': _msg.text});
        _hist.add({'role': 'assistant', 'content': c});
        _msg.clear();
      });
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Negotiation assistant')),
      body: Column(
        children: [
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(12),
              children: [
                TextField(
                  controller: _aid,
                  decoration: const InputDecoration(labelText: 'Optional analysis id'),
                  keyboardType: TextInputType.number,
                ),
                for (final m in _hist)
                  ListTile(
                    title: Text(m['role'] ?? ''),
                    subtitle: Text(m['content'] ?? ''),
                  ),
                if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8),
            child: Row(
              children: [
                Expanded(child: TextField(controller: _msg, decoration: const InputDecoration(hintText: 'Message'))),
                IconButton(onPressed: _busy ? null : _send, icon: const Icon(Icons.send)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// --- 9 Share ---

class ShareFeatureScreen extends StatefulWidget {
  const ShareFeatureScreen({super.key});

  @override
  State<ShareFeatureScreen> createState() => _ShareFeatureScreenState();
}

class _ShareFeatureScreenState extends State<ShareFeatureScreen> {
  final _id = TextEditingController();
  String? _token;
  String? _err;

  @override
  void dispose() {
    _id.dispose();
    super.dispose();
  }

  Future<void> _create() async {
    final i = int.tryParse(_id.text.trim());
    if (i == null) return;
    try {
      final r = await context.read<AppServices>().legato.createShare(i);
      setState(() => _token = r['token']?.toString());
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    }
  }

  Future<void> _copyLink() async {
    if (_token == null) return;
    final link = '${AppConfig.shareBaseUrl}/legato/shares/public/$_token';
    await Clipboard.setData(ClipboardData(text: link));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Share link copied to clipboard')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Share analysis')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(controller: _id, decoration: const InputDecoration(labelText: 'Analysis id'), keyboardType: TextInputType.number),
          FilledButton(onPressed: _create, child: const Text('Create share token')),
          if (_token != null) SelectableText('token: $_token'),
          if (_token != null)
            FilledButton.tonal(onPressed: _copyLink, child: const Text('Copy share link')),
          if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
        ],
      ),
    );
  }
}

// --- 10 Timeline (admin) ---

class TimelineAdminFeatureScreen extends StatefulWidget {
  const TimelineAdminFeatureScreen({super.key});

  @override
  State<TimelineAdminFeatureScreen> createState() => _TimelineAdminFeatureScreenState();
}

class _TimelineAdminFeatureScreenState extends State<TimelineAdminFeatureScreen> {
  bool _busy = false;
  List<dynamic>? _rows;
  String? _err;
  final _aid = TextEditingController();
  final _label = TextEditingController();
  final _date = TextEditingController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  @override
  void dispose() {
    _aid.dispose();
    _label.dispose();
    _date.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    final auth = context.read<AuthProvider>();
    if (auth.user == null || !auth.user!.isAdmin) {
      setState(() => _err = 'Admin only');
      return;
    }
    setState(() {
      _busy = true;
      _err = null;
    });
    try {
      final rows = await context.read<AppServices>().legato.adminTimelineAll();
      setState(() => _rows = rows);
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _add() async {
    final id = int.tryParse(_aid.text.trim());
    if (id == null || _label.text.trim().isEmpty || _date.text.trim().isEmpty) return;
    try {
      await context.read<AppServices>().legato.createTimelineEvent(
            analysisId: id,
            label: _label.text.trim(),
            eventDateIso: _date.text.trim(),
          );
      _label.clear();
      await _load();
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Timeline (admin)')),
      body: _busy
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              children: [
                Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text('Add milestone (owner must own analysis; admin can view all)'),
                      TextField(
                        controller: _aid,
                        decoration: const InputDecoration(labelText: 'Analysis id'),
                        keyboardType: TextInputType.number,
                      ),
                      TextField(controller: _label, decoration: const InputDecoration(labelText: 'Label')),
                      TextField(
                        controller: _date,
                        decoration: const InputDecoration(labelText: 'Date YYYY-MM-DD'),
                      ),
                      FilledButton.tonal(onPressed: _add, child: const Text('Create event')),
                    ],
                  ),
                ),
                if (_err != null)
                  Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  ),
                if (_rows != null)
                  for (final r in _rows!)
                    if (r is Map)
                      ListTile(
                        title: Text('${r['label']}'),
                        subtitle: Text('${r['event_date']} · analysis ${r['analysis_id']}'),
                      ),
              ],
            ),
    );
  }
}

// --- 11 Deal messaging ---

class DealMessagingFeatureScreen extends StatefulWidget {
  const DealMessagingFeatureScreen({super.key});

  @override
  State<DealMessagingFeatureScreen> createState() => _DealMessagingFeatureScreenState();
}

class _DealMessagingFeatureScreenState extends State<DealMessagingFeatureScreen> {
  final _aid = TextEditingController();
  int? _threadId;
  List<dynamic>? _threads;
  final _body = TextEditingController();
  List<dynamic>? _msgs;
  String? _err;

  @override
  void dispose() {
    _aid.dispose();
    _body.dispose();
    super.dispose();
  }

  Future<void> _openThread() async {
    final id = int.tryParse(_aid.text.trim());
    if (id == null) return;
    try {
      final t = await context.read<AppServices>().legato.createDealThread(id, title: 'Discussion');
      final tid = t['id'];
      setState(() {
        _threadId = tid is int ? tid : int.tryParse('$tid');
        _err = null;
      });
      await _loadMsgs();
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    }
  }

  Future<void> _loadThreads() async {
    final id = int.tryParse(_aid.text.trim());
    if (id == null) return;
    try {
      final rows = await context.read<AppServices>().legato.listDealThreads(id);
      if (!mounted) return;
      setState(() => _threads = rows);
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _err = e.message);
    }
  }

  Future<void> _loadMsgs() async {
    if (_threadId == null) return;
    try {
      final m = await context.read<AppServices>().legato.listDealMessages(_threadId!);
      setState(() => _msgs = m);
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    }
  }

  Future<void> _send() async {
    if (_threadId == null || _body.text.trim().isEmpty) return;
    try {
      final text = _body.text.trim();
      _body.clear(); // clear before async gap (prevents disposed-controller crash)
      await context.read<AppServices>().legato.postDealMessage(_threadId!, text);
      await _loadMsgs();
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Deal messaging')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _aid,
                    decoration: const InputDecoration(labelText: 'Analysis id'),
                    keyboardType: TextInputType.number,
                    onSubmitted: (_) => _loadThreads(),
                  ),
                ),
                FilledButton(onPressed: _openThread, child: const Text('Start thread')),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Row(
              children: [
                TextButton(onPressed: _loadThreads, child: const Text('Refresh threads')),
                if (_threads != null) Text('${_threads!.length} threads'),
              ],
            ),
          ),
          if (_threads != null && _threads!.isNotEmpty)
            SizedBox(
              height: 120,
              child: ListView(
                scrollDirection: Axis.horizontal,
                children: [
                  for (final raw in _threads!)
                    if (raw is Map)
                      Padding(
                        padding: const EdgeInsets.only(left: 8),
                        child: ChoiceChip(
                          label: Text('#${raw['id']}'),
                          selected: _threadId == raw['id'],
                          onSelected: (_) async {
                            final tid = raw['id'];
                            setState(() => _threadId = tid is int ? tid : int.tryParse('$tid'));
                            await _loadMsgs();
                          },
                        ),
                      ),
                ],
              ),
            ),
          if (_threadId != null) Text('Thread #$_threadId'),
          Expanded(
            child: ListView(
              children: [
                if (_msgs != null)
                  for (final m in _msgs!)
                    if (m is Map)
                      ListTile(
                        title: Text(m['email']?.toString() ?? ''),
                        subtitle: Text(m['body']?.toString() ?? ''),
                      ),
                if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8),
            child: Row(
              children: [
                Expanded(child: TextField(controller: _body, decoration: const InputDecoration(hintText: 'Message'))),
                IconButton(onPressed: _send, icon: const Icon(Icons.send)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// --- 12 Legal network ---

class LegalNetworkFeatureScreen extends StatefulWidget {
  const LegalNetworkFeatureScreen({super.key});

  @override
  State<LegalNetworkFeatureScreen> createState() => _LegalNetworkFeatureScreenState();
}

class _LegalNetworkFeatureScreenState extends State<LegalNetworkFeatureScreen> {
  final _name = TextEditingController();
  final _headline = TextEditingController();
  final _org = TextEditingController();
  List<dynamic>? _profiles;
  String? _err;

  @override
  void dispose() {
    _name.dispose();
    _headline.dispose();
    _org.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    try {
      await context.read<AppServices>().legato.putLegalProfile({
        'display_name': _name.text,
        'headline': _headline.text,
        'organization': _org.text,
      });
      await _load();
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    }
  }

  Future<void> _load() async {
    try {
      final p = await context.read<AppServices>().legato.listNetworkProfiles();
      setState(() => _profiles = p);
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    }
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      try {
        final me = await context.read<AppServices>().legato.getLegalProfile();
        _name.text = me['display_name']?.toString() ?? '';
        _headline.text = me['headline']?.toString() ?? '';
        _org.text = me['organization']?.toString() ?? '';
        await _load();
      } catch (_) {}
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Legal network (MVP)')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text('MVP: editable profile + directory list. Full “LinkedIn” is a separate product wave.'),
          TextField(controller: _name, decoration: const InputDecoration(labelText: 'Display name')),
          TextField(controller: _headline, decoration: const InputDecoration(labelText: 'Headline')),
          TextField(controller: _org, decoration: const InputDecoration(labelText: 'Organization')),
          FilledButton(onPressed: _save, child: const Text('Save profile')),
          if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          const Divider(),
          const Text('Directory', style: TextStyle(fontWeight: FontWeight.bold)),
          if (_profiles != null)
            for (final p in _profiles!)
              if (p is Map)
                ListTile(
                  title: Text(p['display_name']?.toString() ?? p['email']?.toString() ?? ''),
                  subtitle: Text(p['headline']?.toString() ?? ''),
                ),
        ],
      ),
    );
  }
}
