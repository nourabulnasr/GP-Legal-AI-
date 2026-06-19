import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/history/analysis_detail_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/utils/file_download.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

class ProfileDocumentsScreen extends StatefulWidget {
  const ProfileDocumentsScreen({super.key});

  @override
  State<ProfileDocumentsScreen> createState() => _ProfileDocumentsScreenState();
}

class _ProfileDocumentsScreenState extends State<ProfileDocumentsScreen> {
  bool _loading = true;
  String? _err;
  List<dynamic> _analyses = [];
  Map<String, dynamic>? _lawyerStatus;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final auth = context.read<AuthProvider>();
    final uid = auth.user?.id;
    if (uid == null) return;
    final isLawyer = auth.user?.isLawyerAccount == true;
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final legato = context.read<AppServices>().legato;
      if (isLawyer) {
        final status = await legato.lawyerStatus();
        if (!mounted) return;
        setState(() {
          _lawyerStatus = status;
          _loading = false;
        });
      } else {
        final analyses = await legato.listAnalyses();
        if (!mounted) return;
        setState(() {
          _analyses = analyses;
          _loading = false;
        });
      }
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.message;
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

  Future<void> _openAnalysis(Map<String, dynamic> m) async {
    final id = (m['id'] as num?)?.toInt();
    if (id == null) return;
    final name = m['filename']?.toString() ?? 'Contract';
    final nav = Navigator.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final api = context.read<AppServices>().legato;

    showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (_) => const Center(child: CircularProgressIndicator()),
    );
    try {
      final detail = await api.getAnalysis(id);
      final raw = detail['result_json'];
      final Map<String, dynamic> payload;
      if (raw is String) {
        payload = Map<String, dynamic>.from(jsonDecode(raw) as Map);
      } else if (raw is Map) {
        payload = Map<String, dynamic>.from(raw);
      } else {
        payload = {};
      }
      if (!mounted) return;
      nav.pop();
      await nav.push<void>(
        MaterialPageRoute<void>(
          builder: (_) => AnalysisDetailScreen(
            title: name,
            payload: payload,
            analysisId: id,
          ),
        ),
      );
    } on ApiException catch (e) {
      if (!mounted) return;
      nav.pop();
      messenger.showSnackBar(SnackBar(content: Text(e.message)));
    } catch (e) {
      if (!mounted) return;
      nav.pop();
      messenger.showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  String _mimeFromFilename(String filename) {
    final ext = filename.split('.').last.toLowerCase();
    return switch (ext) {
      'pdf'  => 'application/pdf',
      'png'  => 'image/png',
      'jpg'  => 'image/jpeg',
      'jpeg' => 'image/jpeg',
      'doc'  => 'application/msword',
      'docx' => 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      _      => 'application/octet-stream',
    };
  }

  Future<void> _viewLawyerOwnFile(String fileType, String filename) async {
    final legato = context.read<AppServices>().legato;
    final messenger = ScaffoldMessenger.of(context);
    Uint8List bytes;
    try {
      bytes = switch (fileType) {
        'cv'           => await legato.myLawyerCv(),
        'id-card-back' => await legato.myLawyerIdCardBack(),
        _              => await legato.myLawyerIdCard(),
      };
    } on ApiException catch (e) {
      messenger.showSnackBar(SnackBar(content: Text(e.message)));
      return;
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('$e')));
      return;
    }
    if (!mounted) return;
    final mime = _mimeFromFilename(filename);
    if (mime == 'image/jpeg' || mime == 'image/png') {
      await showDialog<void>(
        context: context,
        builder: (ctx) => Dialog(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AppBar(
                title: Text(filename, overflow: TextOverflow.ellipsis),
                automaticallyImplyLeading: false,
                actions: [
                  IconButton(
                    icon: const Icon(Icons.download_outlined),
                    tooltip: 'Download',
                    onPressed: () => triggerFileDownload(bytes, filename, mime),
                  ),
                  IconButton(icon: const Icon(Icons.close), onPressed: () => Navigator.pop(ctx)),
                ],
              ),
              InteractiveViewer(child: Image.memory(bytes, fit: BoxFit.contain)),
            ],
          ),
        ),
      );
    } else {
      openFileInBrowser(bytes, mime);
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final isLawyer = auth.user?.isLawyerAccount == true;
    return Scaffold(
      appBar: LegatoAppBar(title: const Text('My Documents')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: isLawyer ? _buildLawyerDocs() : _buildAnalyses(),
            ),
    );
  }

  Widget _buildLawyerDocs() {
    final status = _lawyerStatus ?? const <String, dynamic>{};
    final hasCv = status['has_cv'] == true;
    final hasIdCard = status['has_id_card'] == true;
    final hasIdCardBack = status['has_id_card_back'] == true;
    final cvFilename = status['cv_filename']?.toString().isNotEmpty == true
        ? status['cv_filename'].toString()
        : 'cv';
    final idFilename = status['id_card_filename']?.toString().isNotEmpty == true
        ? status['id_card_filename'].toString()
        : 'id_card';
    final idBackFilename = status['id_card_back_filename']?.toString().isNotEmpty == true
        ? status['id_card_back_filename'].toString()
        : 'id_card_back';
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
        Text(
          'Verification documents',
          style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 8),
        if (!hasCv && !hasIdCard && !hasIdCardBack)
          Text(
            'No verification documents on file.',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                ),
          ),
        if (hasCv)
          Card(
            child: ListTile(
              leading: const Icon(Icons.description_outlined, color: LegatoLinkedInTheme.navActiveGold),
              title: const Text('CV / Resume'),
              subtitle: Text(cvFilename, maxLines: 1, overflow: TextOverflow.ellipsis),
              trailing: const Icon(Icons.chevron_right),
              onTap: () => _viewLawyerOwnFile('cv', cvFilename),
            ),
          ),
        if (hasIdCard)
          Card(
            child: ListTile(
              leading: const Icon(Icons.badge_outlined, color: LegatoLinkedInTheme.navActiveGold),
              title: const Text('ID Card (Front)'),
              subtitle: Text(idFilename, maxLines: 1, overflow: TextOverflow.ellipsis),
              trailing: const Icon(Icons.chevron_right),
              onTap: () => _viewLawyerOwnFile('id-card', idFilename),
            ),
          ),
        if (hasIdCardBack)
          Card(
            child: ListTile(
              leading: const Icon(Icons.badge_outlined, color: LegatoLinkedInTheme.navActiveGold),
              title: const Text('ID Card (Back)'),
              subtitle: Text(idBackFilename, maxLines: 1, overflow: TextOverflow.ellipsis),
              trailing: const Icon(Icons.chevron_right),
              onTap: () => _viewLawyerOwnFile('id-card-back', idBackFilename),
            ),
          ),
      ],
    );
  }

  Widget _buildAnalyses() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
        Text(
          'Analyzed contracts (your uploads)',
          style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 8),
        if (_analyses.isEmpty)
          Text(
            'No saved analyses yet. Use Analyze contract from Home.',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                ),
          )
        else
          ..._analyses.map((raw) {
            final m = Map<String, dynamic>.from(raw as Map);
            return Card(
              child: ListTile(
                leading: const Icon(Icons.description_outlined, color: LegatoLinkedInTheme.navActiveGold),
                title: Text(m['filename']?.toString() ?? 'Contract'),
                subtitle: Text(
                  m['created_at']?.toString() ?? '',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => _openAnalysis(m),
              ),
            );
          }),
      ],
    );
  }
}
