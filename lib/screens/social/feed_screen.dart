import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/config/app_config.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/chat/chat_hub_screen.dart';
import 'package:legato_mobile/screens/social/member_profile_screen.dart';
import 'package:legato_mobile/screens/social/social_constants.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/utils/platform_file_bytes.dart';
import 'package:legato_mobile/widgets/user_avatar.dart';

String _relativeTime(String iso) {
  try {
    // Parse the ISO string as UTC (keeps Z), then convert to device local time.
    final dt = DateTime.parse(iso).toLocal();
    final d = DateTime.now().difference(dt);
    if (d.inSeconds < 60) return '${d.inSeconds}s ago';
    if (d.inMinutes < 60) return '${d.inMinutes}m ago';
    if (d.inHours < 48) return '${d.inHours}h ago';
    if (d.inDays < 14) return '${d.inDays}d ago';
    return '${dt.day}/${dt.month}/${dt.year}';
  } catch (_) {
    return iso;
  }
}

Future<bool> _launchExternalUri(Uri uri) async {
  try {
    final mode = kIsWeb ? LaunchMode.platformDefault : LaunchMode.externalApplication;
    return await launchUrl(uri, mode: mode);
  } catch (_) {
    return false;
  }
}

Future<void> _shareViaWhatsApp(BuildContext context, String message) async {
  final webUri = Uri.https('wa.me', '', {'text': message});
  if (await _launchExternalUri(webUri)) return;

  if (!kIsWeb) {
    final appUri = Uri(scheme: 'whatsapp', host: 'send', queryParameters: {'text': message});
    if (await _launchExternalUri(appUri)) return;
  }

  await Clipboard.setData(ClipboardData(text: message));
  if (context.mounted) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Could not open WhatsApp. Message copied to clipboard — paste it in WhatsApp.'),
      ),
    );
  }
}

class FeedScreen extends StatefulWidget {
  const FeedScreen({super.key});

  @override
  State<FeedScreen> createState() => _FeedScreenState();
}

class _FeedScreenState extends State<FeedScreen> {
  final _scroll = ScrollController();
  String _category = 'All Updates';
  final List<Map<String, dynamic>> _posts = [];
  int _page = 1;
  bool _loading = false;
  bool _hasMore = true;
  String? _err;
  String? _myAvatarUrl;
  String _myDisplayName = '';

  @override
  void initState() {
    super.initState();
    _scroll.addListener(_onScroll);
    _load(reset: true);
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadMyAvatar());
  }

  Future<void> _loadMyAvatar() async {
    final auth = context.read<AuthProvider>();
    final uid = auth.user?.id;
    final email = auth.user?.email ?? '';
    if (uid == null) return;
    try {
      final d = await context.read<AppServices>().legato.getSocialProfileResilient(uid, email);
      if (!mounted) return;
      setState(() {
        _myAvatarUrl = d['avatar_url']?.toString();
        _myDisplayName = d['display_name']?.toString() ??
            (email.contains('@') ? email.split('@').first : email);
      });
    } catch (_) {}
  }

  @override
  void dispose() {
    _scroll.removeListener(_onScroll);
    _scroll.dispose();
    super.dispose();
  }

  void _onScroll() {
    if (!_hasMore || _loading) return;
    if (_scroll.position.pixels > _scroll.position.maxScrollExtent - 280) {
      _load();
    }
  }

  Future<void> _load({bool reset = false}) async {
    if (_loading) return;
    setState(() {
      _loading = true;
      if (reset) {
        _err = null;
        _page = 1;
        _posts.clear();
        _hasMore = true;
      }
    });
    try {
      final api = context.read<AppServices>().legato;
      final res = await api.getPosts(
        category: _category == 'All Updates' ? null : _category,
        page: _page,
      );
      final items = (res['items'] as List<dynamic>?) ?? [];
      final total = (res['total'] as num?)?.toInt() ?? 0;
      if (!mounted) return;
      setState(() {
        for (final raw in items) {
          if (raw is Map<String, dynamic>) {
            _posts.add(raw);
          } else if (raw is Map) {
            _posts.add(Map<String, dynamic>.from(raw));
          }
        }
        _page++;
        _hasMore = _posts.length < total;
        _loading = false;
      });
    } on ApiException catch (e) {
      if (mounted) {
        setState(() {
          _err = e.message;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _err = e.toString();
          _loading = false;
        });
      }
    }
  }

  Future<void> _openComposer() async {
    final initial = _category == 'All Updates' ? 'All Updates' : _category;
    final app = context.read<AppServices>();
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => _FeedComposerSheet(
        initialCategory: initial,
        app: app,
        onPosted: () => _load(reset: true),
      ),
    );
    if (mounted) _loadMyAvatar();
  }

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: Theme.of(context).scaffoldBackgroundColor,
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
              child: Row(
                children: [
                  Container(
                    width: 36,
                    height: 36,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color: const Color(0xFF1B1F23),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      'L',
                      style: TextStyle(
                        color: LegatoLinkedInTheme.navActiveGold,
                        fontWeight: FontWeight.w800,
                        fontSize: 18,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    'Legato',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.w700,
                          fontFamily: 'serif',
                        ),
                  ),
                  const Spacer(),
                  IconButton(
                    tooltip: 'Messages',
                    onPressed: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(builder: (_) => const ChatHubScreen()),
                    ),
                    icon: const Icon(Icons.chat_bubble_outline),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            SizedBox(
              height: 40,
              child: ListView(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 12),
                children: kFeedCategoryFilters.map((c) {
                  final sel = _category == c;
                  return Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: FilterChip(
                      label: Text(c),
                      selected: sel,
                      onSelected: (_) async {
                        setState(() => _category = c);
                        await _load(reset: true);
                      },
                      selectedColor: const Color(0xFF1B1F23),
                      labelStyle: TextStyle(color: sel ? Colors.white : LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                    ),
                  );
                }).toList(),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
              child: Material(
                color: Theme.of(context).colorScheme.surface,
                borderRadius: BorderRadius.circular(8),
                child: InkWell(
                  onTap: _openComposer,
                  borderRadius: BorderRadius.circular(8),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
                    child: Row(
                      children: [
                        UserAvatar(
                          radius: 18,
                          imageUrl: _myAvatarUrl,
                          name: _myDisplayName,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'Share an update or insight…',
                            style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                          ),
                        ),
                        Icon(Icons.add_circle, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            if (_err != null)
              Padding(
                padding: const EdgeInsets.all(8),
                child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error, fontSize: 13)),
              ),
            Expanded(
              child: RefreshIndicator(
                onRefresh: () => _load(reset: true),
                child: _posts.isEmpty && !_loading
                    ? ListView(
                        children: const [
                          SizedBox(height: 48),
                          Center(child: Text('No posts yet. Create the first one.')),
                        ],
                      )
                    : ListView.builder(
                        controller: _scroll,
                        padding: const EdgeInsets.fromLTRB(12, 0, 12, 24),
                        itemCount: _posts.length + (_hasMore ? 1 : 0),
                        itemBuilder: (context, i) {
                          if (i >= _posts.length) {
                            return const Padding(
                              padding: EdgeInsets.all(16),
                              child: Center(child: CircularProgressIndicator()),
                            );
                          }
                          return _PostCard(
                            post: _posts[i],
                            onChanged: () => _load(reset: true),
                          );
                        },
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Owns [TextEditingController] so it is disposed in [State.dispose] after the sheet route is
/// unmounted — avoids `'_dependents.isEmpty': is not true` from disposing a controller in
/// [Future.whenComplete] while the sheet subtree is still tearing down.
class _FeedComposerSheet extends StatefulWidget {
  const _FeedComposerSheet({
    required this.initialCategory,
    required this.app,
    required this.onPosted,
  });

  final String initialCategory;
  final AppServices app;
  final Future<void> Function() onPosted;

  @override
  State<_FeedComposerSheet> createState() => _FeedComposerSheetState();
}

class _FeedComposerSheetState extends State<_FeedComposerSheet> {
  late final TextEditingController _textCtrl;
  late String _category;
  final Set<String> _tags = {};
  Uint8List? _imageBytes;
  String? _imageFilename;
  bool _posting = false;

  @override
  void initState() {
    super.initState();
    _textCtrl = TextEditingController();
    _category = widget.initialCategory;
  }

  @override
  void dispose() {
    _textCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true,
      allowMultiple: false,
    );
    if (result == null || result.files.isEmpty) return;
    final file = result.files.first;
    final bytes = await readPlatformFileBytes(file);
    if (bytes == null || bytes.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not read the selected photo. Try another image.')),
        );
      }
      return;
    }
    if (mounted) {
      setState(() {
        _imageBytes = bytes;
        _imageFilename = pickedImageFilename(file);
      });
    }
  }

  void _removeImage() => setState(() {
        _imageBytes = null;
        _imageFilename = null;
      });

  Future<void> _submit() async {
    final body = _textCtrl.text.trim();
    if (body.isEmpty && _imageBytes == null) return;
    setState(() => _posting = true);
    try {
      await widget.app.legato.createPost(
        content: body,
        tags: _tags.toList(),
        category: _category,
        imageBytes: _imageBytes,
        imageFilename: _imageFilename,
      );
      if (!mounted) return;
      final posted = widget.onPosted;
      Navigator.of(context).pop();
      await posted();
    } on ApiException catch (e) {
      if (mounted) {
        setState(() => _posting = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      }
    } catch (e) {
      if (mounted) {
        setState(() => _posting = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(context).bottom),
      child: Material(
        color: Theme.of(context).scaffoldBackgroundColor,
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'New post',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _textCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Share an update or insight…',
                    alignLabelWithHint: true,
                  ),
                  minLines: 4,
                  maxLines: 10,
                ),
                const SizedBox(height: 8),
                // Image attach row
                Row(
                  children: [
                    TextButton.icon(
                      onPressed: _posting ? null : _pickImage,
                      icon: const Icon(Icons.photo_outlined, size: 20),
                      label: const Text('Photo'),
                    ),
                  ],
                ),
                // Image preview
                if (_imageBytes != null) ...[
                  const SizedBox(height: 8),
                  Stack(
                    alignment: Alignment.topRight,
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.memory(
                          _imageBytes!,
                          height: 180,
                          width: double.infinity,
                          fit: BoxFit.cover,
                        ),
                      ),
                      Positioned(
                        top: 4,
                        right: 4,
                        child: Material(
                          color: Colors.black54,
                          shape: const CircleBorder(),
                          child: InkWell(
                            customBorder: const CircleBorder(),
                            onTap: _removeImage,
                            child: const Padding(
                              padding: EdgeInsets.all(6),
                              child: Icon(Icons.close, size: 20, color: Colors.white),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      onPressed: _posting ? null : _removeImage,
                      icon: const Icon(Icons.delete_outline, size: 18),
                      label: const Text('Remove photo'),
                    ),
                  ),
                ],
                const SizedBox(height: 12),
                Text('Category', style: Theme.of(context).textTheme.labelLarge),
                const SizedBox(height: 6),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: kFeedCategoryFilters.map((c) {
                    final sel = _category == c;
                    return FilterChip(
                      label: Text(c),
                      selected: sel,
                      onSelected: (_) => setState(() => _category = c),
                    );
                  }).toList(),
                ),
                const SizedBox(height: 12),
                Text('Tags', style: Theme.of(context).textTheme.labelLarge),
                const SizedBox(height: 6),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: kLegalTopicTags.map((t) {
                    final on = _tags.contains(t);
                    return FilterChip(
                      label: Text(t),
                      selected: on,
                      onSelected: (v) {
                        setState(() {
                          if (v) {
                            _tags.add(t);
                          } else {
                            _tags.remove(t);
                          }
                        });
                      },
                    );
                  }).toList(),
                ),
                const SizedBox(height: 20),
                FilledButton(
                  onPressed: _posting ? null : _submit,
                  child: _posting
                      ? const SizedBox(
                          height: 18,
                          width: 18,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Text('Post'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _PostCard extends StatefulWidget {
  const _PostCard({required this.post, required this.onChanged});

  final Map<String, dynamic> post;
  final VoidCallback onChanged;

  @override
  State<_PostCard> createState() => _PostCardState();
}

class _PostCardState extends State<_PostCard> {
  bool _expanded = false;
  bool _loadingComments = false;
  String? _commentsErr;
  List<Map<String, dynamic>> _comments = [];

  Future<void> _toggleLike() async {
    final id = widget.post['id'] as int?;
    if (id == null) return;
    try {
      final r = await context.read<AppServices>().legato.togglePostLike(id);
      setState(() {
        widget.post['liked'] = r['liked'];
        widget.post['likes_count'] = r['likes_count'];
      });
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  Future<void> _showShareSheet() async {
    final id = widget.post['id'] as int?;
    if (id == null) return;

    final link = '${AppConfig.shareBaseUrl}/?post=$id';
    final shareMessage = 'Check out this post on Legato: $link';

    await showModalBottomSheet<void>(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.only(top: 12, bottom: 8),
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text('Share post', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
              ),
            ),
            ListTile(
              leading: const Icon(Icons.link_outlined),
              title: const Text('Copy link'),
              onTap: () async {
                await Clipboard.setData(ClipboardData(text: link));
                if (!ctx.mounted) return;
                Navigator.pop(ctx);
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Link copied to clipboard')),
                  );
                }
                await _recordShare(id);
              },
            ),
            ListTile(
              leading: const Icon(Icons.open_in_browser_outlined),
              title: const Text('Open in browser'),
              onTap: () async {
                Navigator.pop(ctx);
                final opened = await _launchExternalUri(Uri.parse(link));
                if (!opened && mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Could not open link in browser')),
                  );
                }
                await _recordShare(id);
              },
            ),
            ListTile(
              leading: const Icon(Icons.chat_outlined, color: Color(0xFF25D366)),
              title: const Text('WhatsApp'),
              onTap: () async {
                Navigator.pop(ctx);
                await _shareViaWhatsApp(context, shareMessage);
                await _recordShare(id);
              },
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  Future<void> _recordShare(int id) async {
    final prev = (widget.post['shares_count'] as num?)?.toInt() ?? 0;
    setState(() => widget.post['shares_count'] = prev + 1);
    try {
      final r = await context.read<AppServices>().legato.sharePost(id);
      if (!mounted) return;
      setState(() => widget.post['shares_count'] = r['shares_count'] ?? (prev + 1));
    } on ApiException catch (_) {
      if (!mounted) return;
      setState(() => widget.post['shares_count'] = prev);
    } catch (_) {
      if (!mounted) return;
      setState(() => widget.post['shares_count'] = prev);
    }
  }

  Future<void> _loadComments() async {
    final id = widget.post['id'] as int?;
    if (id == null) return;
    setState(() {
      _loadingComments = true;
      _commentsErr = null;
    });
    try {
      final r = await context.read<AppServices>().legato.getPostComments(id);
      // Handle both 'items' and 'comments' response keys.
      final raw = (r['items'] as List<dynamic>?) ?? (r['comments'] as List<dynamic>?) ?? [];
      if (!mounted) return;
      setState(() {
        _comments = raw.map((e) => Map<String, dynamic>.from(e as Map)).toList();
        _loadingComments = false;
      });
    } on ApiException catch (e) {
      if (mounted) {
        setState(() {
          _commentsErr = e.message;
          _loadingComments = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _commentsErr = e.toString();
          _loadingComments = false;
        });
      }
    }
  }

  Future<void> _addComment(String text) async {
    final id = widget.post['id'] as int?;
    if (id == null || text.trim().isEmpty) return;
    try {
      await context.read<AppServices>().legato.addPostComment(id, text.trim());
      widget.onChanged();
      await _loadComments();
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  void _goToProfile() {
    final p = widget.post;
    final authorId = (p['author_id'] as num?)?.toInt() ?? (p['user_id'] as num?)?.toInt();
    if (authorId == null || authorId == 0) return;
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => MemberProfileScreen(userId: authorId)),
    );
  }

  Future<void> _deletePost() async {
    final id = widget.post['id'] as int?;
    if (id == null) return;
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete post'),
        content: const Text('Delete this post permanently?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
    if (confirm != true) return;
    try {
      await context.read<AppServices>().legato.deletePost(id);
      widget.onChanged();
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final p = widget.post;
    final tags = (p['tags'] as List<dynamic>?) ?? [];
    final lc = (p['likes_count'] as num?)?.toInt() ?? 0;
    final cc = (p['comments_count'] as num?)?.toInt() ?? 0;
    final sc = (p['shares_count'] as num?)?.toInt() ?? 0;
    final liked = p['liked'] == true;
    final meId = context.watch<AuthProvider>().user?.id;
    final authorId = (p['author_id'] as num?)?.toInt() ?? (p['user_id'] as num?)?.toInt();
    final isOwn = meId != null && authorId != null && meId == authorId;

    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Author row — tappable → profile
            GestureDetector(
              onTap: _goToProfile,
              behavior: HitTestBehavior.opaque,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  UserAvatar(
                    radius: 22,
                    imageUrl: p['author_avatar_url']?.toString(),
                    name: p['author_name']?.toString() ?? 'Member',
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          p['author_name']?.toString() ?? 'Member',
                          style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700),
                        ),
                        Text(
                          p['author_subtitle']?.toString() ?? '',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        Text(
                          _relativeTime(p['created_at']?.toString() ?? ''),
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context), fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                  if (isOwn)
                    PopupMenuButton<String>(
                      icon: const Icon(Icons.more_horiz, size: 20),
                      onSelected: (v) {
                        if (v == 'delete') _deletePost();
                      },
                      itemBuilder: (_) => [
                        const PopupMenuItem(value: 'delete', child: Text('Delete post', style: TextStyle(color: Colors.red))),
                      ],
                    ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            SelectableText(p['content']?.toString() ?? '', style: Theme.of(context).textTheme.bodyMedium),
            if ((p['image_url'] as String?)?.isNotEmpty == true) ...[
              const SizedBox(height: 10),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.network(
                  p['image_url'] as String,
                  width: double.infinity,
                  height: 240,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, _) => const SizedBox.shrink(),
                ),
              ),
            ],
            if (tags.isNotEmpty) ...[
              const SizedBox(height: 10),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: tags.map<Widget>((t) {
                  return Chip(
                    label: Text(t.toString(), style: const TextStyle(fontSize: 12)),
                    visualDensity: VisualDensity.compact,
                    padding: EdgeInsets.zero,
                  );
                }).toList(),
              ),
            ],
            const SizedBox(height: 8),
            Text(
              '$lc likes · $cc comments · $sc shares',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
            ),
            const Divider(height: 20),
            Row(
              children: [
                TextButton.icon(
                  onPressed: _toggleLike,
                  icon: Icon(Icons.thumb_up_alt_outlined, size: 18, color: liked ? LegatoLinkedInTheme.navActiveGold : null),
                  label: const Text('Like'),
                ),
                TextButton.icon(
                  onPressed: () async {
                    setState(() => _expanded = !_expanded);
                    if (_expanded) await _loadComments();
                  },
                  icon: const Icon(Icons.chat_bubble_outline, size: 18),
                  label: const Text('Comment'),
                ),
                TextButton.icon(
                  onPressed: _showShareSheet,
                  icon: const Icon(Icons.share_outlined, size: 18),
                  label: const Text('Share'),
                ),
              ],
            ),
            if (_expanded) ...[
              if (_loadingComments)
                const LinearProgressIndicator(minHeight: 2)
              else if (_commentsErr != null)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Text(_commentsErr!, style: TextStyle(color: Theme.of(context).colorScheme.error, fontSize: 12)),
                )
              else if (_comments.isEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  child: Text('No comments yet. Be the first!', style: TextStyle(fontSize: 13, color: LegatoLinkedInTheme.textSecondaryAdaptive(context))),
                ),
              ..._comments.map(
                (c) => ListTile(
                  dense: true,
                  leading: UserAvatar(
                    radius: 16,
                    imageUrl: c['author_avatar_url']?.toString(),
                    name: c['author_name']?.toString() ?? '',
                  ),
                  title: Text(c['author_name']?.toString() ?? ''),
                  subtitle: Text(c['content']?.toString() ?? ''),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: _CommentField(onSubmit: _addComment),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _CommentField extends StatefulWidget {
  const _CommentField({required this.onSubmit});

  final Future<void> Function(String) onSubmit;

  @override
  State<_CommentField> createState() => _CommentFieldState();
}

class _CommentFieldState extends State<_CommentField> {
  final _c = TextEditingController();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _c,
            decoration: const InputDecoration(
              hintText: 'Add a comment…',
              isDense: true,
            ),
            minLines: 1,
            maxLines: 3,
          ),
        ),
        IconButton(
          onPressed: () async {
            final text = _c.text.trim();
            if (text.isEmpty) return;
            // Clear BEFORE the async gap so we never touch the controller after dispose.
            _c.clear();
            await widget.onSubmit(text);
          },
          icon: const Icon(Icons.send),
        ),
      ],
    );
  }
}
