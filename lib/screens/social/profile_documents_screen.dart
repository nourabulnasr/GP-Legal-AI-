import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class ProfileDocumentsScreen extends StatefulWidget {
  const ProfileDocumentsScreen({super.key});

  @override
  State<ProfileDocumentsScreen> createState() => _ProfileDocumentsScreenState();
}

class _ProfileDocumentsScreenState extends State<ProfileDocumentsScreen> {
  bool _loading = true;
  String? _err;
  List<dynamic> _items = [];
  List<dynamic> _analyses = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final uid = context.read<AuthProvider>().user?.id;
    if (uid == null) return;
    setState(() {
      _loading = true;
      _err = null;
    });
    final app = context.read<AppServices>();
    final errs = <String>[];
    var items = <dynamic>[];
    var analyses = <dynamic>[];
    try {
      final r = await app.legato.getProfileDocuments(uid);
      items = (r['items'] as List<dynamic>?) ?? [];
    } on ApiException catch (e) {
      errs.add('Links: ${e.message}');
    } catch (e) {
      errs.add('Links: $e');
    }
    try {
      analyses = await app.legato.listAnalyses();
    } on ApiException catch (e) {
      errs.add('Analyses: ${e.message}');
    } catch (e) {
      errs.add('Analyses: $e');
    }
    if (!mounted) return;
    setState(() {
      _items = items;
      _analyses = analyses;
      _err = errs.isEmpty ? null : errs.join('\n');
      _loading = false;
    });
  }

  Future<void> _add() async {
    final title = TextEditingController();
    final url = TextEditingController();
    final app = context.read<AppServices>();
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add document'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(controller: title, decoration: const InputDecoration(labelText: 'Title')),
            TextField(controller: url, decoration: const InputDecoration(labelText: 'File URL')),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: LegatoLinkedInTheme.navActiveGold,
              foregroundColor: const Color(0xFF1B1F23),
            ),
            onPressed: () async {
              final t = title.text.trim();
              final u = url.text.trim();
              if (t.isEmpty || u.isEmpty) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Enter a title and file URL')),
                );
                return;
              }
              try {
                await app.legato.addProfileDocument(title: t, fileUrl: u);
                if (!mounted) return;
                if (ctx.mounted) Navigator.pop(ctx);
                await _load();
              } on ApiException catch (e) {
                if (ctx.mounted) ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(content: Text(e.message)));
              } catch (e) {
                if (ctx.mounted) ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(content: Text('$e')));
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    title.dispose();
    url.dispose();
  }

  Future<void> _upload() async {
    final app = context.read<AppServices>();
    final messenger = ScaffoldMessenger.of(context);
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['png', 'jpg', 'jpeg', 'webp', 'pdf'],
      withData: false,
    );
    if (result == null || result.files.isEmpty) return;
    final f = result.files.single;
    if (f.path == null) return;
    final title = TextEditingController(text: f.name);
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Upload document'),
        content: TextField(controller: title, decoration: const InputDecoration(labelText: 'Title')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: LegatoLinkedInTheme.navActiveGold,
              foregroundColor: const Color(0xFF1B1F23),
            ),
            onPressed: () async {
              try {
                await app.legato.uploadProfileDocument(
                  title: title.text.trim().isEmpty ? f.name : title.text.trim(),
                  filePath: f.path!,
                  filename: f.name,
                );
                if (!mounted) return;
                if (ctx.mounted) Navigator.pop(ctx);
                await _load();
              } on ApiException catch (e) {
                if (ctx.mounted) messenger.showSnackBar(SnackBar(content: Text(e.message)));
              } catch (e) {
                if (ctx.mounted) messenger.showSnackBar(SnackBar(content: Text('$e')));
              }
            },
            child: const Text('Upload'),
          ),
        ],
      ),
    );
    title.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('My Documents')),
      floatingActionButton: FloatingActionButton(
        backgroundColor: LegatoLinkedInTheme.navActiveGold,
        foregroundColor: const Color(0xFF1B1F23),
        onPressed: () async {
          final choice = await showModalBottomSheet<String>(
            context: context,
            builder: (ctx) => SafeArea(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ListTile(
                    leading: const Icon(Icons.upload_file_outlined),
                    title: const Text('Upload image/PDF (saved in DB)'),
                    onTap: () => Navigator.pop(ctx, 'upload'),
                  ),
                  ListTile(
                    leading: const Icon(Icons.link),
                    title: const Text('Add link (URL)'),
                    onTap: () => Navigator.pop(ctx, 'link'),
                  ),
                ],
              ),
            ),
          );
          if (choice == 'upload') return _upload();
          if (choice == 'link') return _add();
        },
        child: const Icon(Icons.add),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_err != null) Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  Text(
                    'Analyzed contracts (your uploads)',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_analyses.isEmpty)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 16),
                      child: Text(
                        'No saved analyses yet. Use Contracts → Analyze contract.',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                      ),
                    )
                  else
                    ..._analyses.map((raw) {
                      final m = raw as Map<String, dynamic>;
                      return Card(
                        child: ListTile(
                          leading: const Icon(Icons.description_outlined, color: LegatoLinkedInTheme.navActiveGold),
                          title: Text(m['filename']?.toString() ?? 'Contract'),
                          subtitle: Text(m['created_at']?.toString() ?? '', maxLines: 1, overflow: TextOverflow.ellipsis),
                        ),
                      );
                    }),
                  const SizedBox(height: 8),
                  Text(
                    'Profile links',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  if (_items.isEmpty)
                    Text(
                      'No profile links yet. Tap + to add a title and URL.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                    )
                  else
                    ..._items.map((raw) {
                      final m = raw as Map<String, dynamic>;
                      return Card(
                        child: ListTile(
                          leading: const Icon(Icons.link, color: LegatoLinkedInTheme.navActiveGold),
                          title: Text(m['title']?.toString() ?? ''),
                          subtitle: Text(m['file_url']?.toString() ?? '', maxLines: 2, overflow: TextOverflow.ellipsis),
                        ),
                      );
                    }),
                ],
              ),
            ),
    );
  }
}
