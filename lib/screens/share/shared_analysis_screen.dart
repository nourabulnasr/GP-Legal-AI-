import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/screens/auth_gate.dart';
import 'package:legato_mobile/screens/history/analysis_detail_screen.dart';
import 'package:legato_mobile/utils/share_link.dart';
import 'package:legato_mobile/utils/share_link_url.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

/// Read-only analysis opened from a public share link.
class SharedAnalysisScreen extends StatefulWidget {
  const SharedAnalysisScreen({
    super.key,
    required this.token,
    this.onDismiss,
  });

  final String token;
  final VoidCallback? onDismiss;

  @override
  State<SharedAnalysisScreen> createState() => _SharedAnalysisScreenState();
}

class _SharedAnalysisScreenState extends State<SharedAnalysisScreen> {
  bool _loading = true;
  String? _err;
  String _title = 'Shared analysis';
  Map<String, dynamic>? _payload;
  int? _analysisId;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final data = await context.read<AppServices>().legato.publicShare(widget.token);
      if (!mounted) return;
      final result = data['result'];
      final payload = result is Map<String, dynamic>
          ? result
          : result is Map
              ? Map<String, dynamic>.from(result)
              : <String, dynamic>{};
      setState(() {
        _title = data['filename']?.toString() ?? 'Shared analysis';
        _analysisId = (data['analysis_id'] as num?)?.toInt();
        _payload = payload;
        _loading = false;
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.statusCode == 410 ? 'This share link has expired.' : e.message;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.toString();
        _loading = false;
      });
    }
  }

  void _dismiss() {
    if (kIsWeb) clearShareQueryFromUrl();
    final navigator = Navigator.of(context);
    if (navigator.canPop()) {
      navigator.pop();
    } else {
      widget.onDismiss?.call();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return Scaffold(
        appBar: LegatoAppBar(
          title: const Text('Shared analysis'),
          leading: IconButton(icon: const Icon(Icons.close), onPressed: _dismiss),
        ),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    if (_err != null || _payload == null) {
      return Scaffold(
        appBar: LegatoAppBar(
          title: const Text('Shared analysis'),
          leading: IconButton(icon: const Icon(Icons.close), onPressed: _dismiss),
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.link_off, size: 48, color: Theme.of(context).colorScheme.error),
                const SizedBox(height: 16),
                Text(
                  _err ?? 'Could not load shared analysis.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
                const SizedBox(height: 16),
                FilledButton(onPressed: _load, child: const Text('Retry')),
              ],
            ),
          ),
        ),
      );
    }

    return AnalysisDetailScreen(
      title: _title,
      payload: _payload!,
      analysisId: _analysisId,
      readOnly: true,
      onClose: _dismiss,
    );
  }
}

/// Opens shared analysis when the URL contains a share token; otherwise auth flow.
class ShareLinkGate extends StatefulWidget {
  const ShareLinkGate({super.key});

  @override
  State<ShareLinkGate> createState() => _ShareLinkGateState();
}

class _ShareLinkGateState extends State<ShareLinkGate> {
  String? _shareToken;

  @override
  void initState() {
    super.initState();
    _shareToken = parseShareTokenFromUri(Uri.base);
  }

  void _clearShare() {
    setState(() => _shareToken = null);
    if (kIsWeb) clearShareQueryFromUrl();
  }

  @override
  Widget build(BuildContext context) {
    final token = _shareToken;
    if (token != null && token.isNotEmpty) {
      return SharedAnalysisScreen(token: token, onDismiss: _clearShare);
    }
    return const AuthGate();
  }
}
