import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/utils/local_lawyer_queue.dart';
import 'package:legato_mobile/utils/pending_lawyer_docs.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

class LawyerApplicationScreen extends StatefulWidget {
  const LawyerApplicationScreen({super.key});

  @override
  State<LawyerApplicationScreen> createState() => _LawyerApplicationScreenState();
}

class _LawyerApplicationScreenState extends State<LawyerApplicationScreen> {
  final _licenseController = TextEditingController();
  final _yearsController = TextEditingController();

  bool _loadingStatus = true;
  bool _submitting = false;
  String? _err;
  String? _successMsg;

  Map<String, dynamic>? _statusData;

  Uint8List? _cvBytes;
  String? _cvFilename;
  Uint8List? _idCardBytes;
  String? _idCardFilename;

  // Tracks files already on the server so re-upload is optional
  bool _hasServerCv = false;
  bool _hasServerIdCard = false;
  String? _serverCvFilename;
  String? _serverIdCardFilename;

  @override
  void initState() {
    super.initState();
    // Pre-populate docs stashed during registration (old-server fallback)
    if (PendingLawyerDocs.hasPending) {
      _cvBytes = PendingLawyerDocs.cvBytes;
      _cvFilename = PendingLawyerDocs.cvFilename;
      _idCardBytes = PendingLawyerDocs.idCardBytes;
      _idCardFilename = PendingLawyerDocs.idCardFilename;
      PendingLawyerDocs.clear();
    }
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadStatus());
  }

  @override
  void dispose() {
    _licenseController.dispose();
    _yearsController.dispose();
    super.dispose();
  }

  Future<void> _loadStatus() async {
    setState(() {
      _loadingStatus = true;
      _err = null;
    });
    final legato = context.read<AppServices>().legato;
    final auth = context.read<AuthProvider>();
    try {
      final data = await legato.lawyerStatus();
      if (!mounted) return;
      setState(() {
        _statusData = data;
        _hasServerCv = data['has_cv'] == true;
        _hasServerIdCard = data['has_id_card'] == true;
        _serverCvFilename = data['cv_filename']?.toString();
        _serverIdCardFilename = data['id_card_filename']?.toString();
        // Pre-populate editable fields so rejected users can adjust and resubmit
        final status = data['status']?.toString() ?? '';
        if (status == 'rejected' || status == 'not_applied') {
          final lic = data['bar_license_number']?.toString() ?? '';
          if (lic.isNotEmpty) _licenseController.text = lic;
          final yoe = data['years_of_experience'];
          if (yoe != null) _yearsController.text = yoe.toString();
        }
      });
      if (data['status'] == 'approved') {
        await auth.refreshUser();
      }
    } on ApiException catch (e) {
      if (!mounted) return;
      if (e.statusCode == 404) {
        // Endpoint not deployed — check the local queue for an existing submission.
        Map<String, dynamic>? localData;
        try {
          final userId = auth.user?.id;
          if (userId != null) {
            final all = await LocalLawyerQueue.getAll();
            for (final entry in all) {
              if ((entry['user_id'] as int?) == userId) { localData = entry; break; }
            }
          }
        } catch (_) {}
        if (mounted) setState(() => _statusData = localData ?? {'status': 'not_applied'});
      } else {
        setState(() => _err = e.message);
      }
    } catch (e) {
      if (mounted) setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _loadingStatus = false);
    }
  }

  Future<void> _pickCv() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'doc', 'docx', 'jpg', 'jpeg', 'png'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;
    final file = result.files.first;
    if (file.bytes == null) return;
    setState(() {
      _cvBytes = file.bytes;
      _cvFilename = file.name;
    });
  }

  Future<void> _pickIdCard() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['jpg', 'jpeg', 'png', 'pdf'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;
    final file = result.files.first;
    if (file.bytes == null) return;
    setState(() {
      _idCardBytes = file.bytes;
      _idCardFilename = file.name;
    });
  }

  Future<void> _submit() async {
    if (_cvBytes == null && !_hasServerCv) {
      setState(() => _err = 'Please select your CV / resume.');
      return;
    }
    if (_idCardBytes == null && !_hasServerIdCard) {
      setState(() => _err = 'Please select your national ID card.');
      return;
    }
    setState(() {
      _submitting = true;
      _err = null;
      _successMsg = null;
    });
    final legato = context.read<AppServices>().legato;
    final auth = context.read<AuthProvider>();
    try {
      await legato.lawyerApply(
        barLicenseNumber: _licenseController.text.trim(),
        yearsOfExperience: int.tryParse(_yearsController.text.trim()),
        cvBytes: _cvBytes,
        cvFilename: _cvFilename,
        idCardBytes: _idCardBytes,
        idCardFilename: _idCardFilename,
      );
      PendingLawyerDocs.clear();
      setState(() => _successMsg = 'Application submitted! An admin will review your documents.');
      await _loadStatus();
      await auth.refreshUser();
    } on ApiException catch (e) {
      if (e.statusCode == 404) {
        // /lawyer/apply is not deployed yet — store the application locally so
        // the admin can see and approve it from the same device/browser.
        try {
          final u = auth.user;
          if (u != null) {
            await LocalLawyerQueue.addOrUpdate(
              userId: u.id,
              email: u.email,
              cvFilename: _cvFilename,
              cvBytes: _cvBytes,
              idCardFilename: _idCardFilename,
              idCardBytes: _idCardBytes,
              yearsOfExperience: int.tryParse(_yearsController.text.trim()),
              barLicenseNumber: _licenseController.text.trim(),
            );
          }
          PendingLawyerDocs.clear();
          setState(() => _successMsg = 'Application submitted! An admin will review your documents.');
          await _loadStatus();
          await auth.refreshUser();
        } catch (_) {
          setState(() => _err =
              'Could not save your application. Please try again.');
        }
      } else if (e.statusCode == 400 &&
          (e.message.contains('lawyer') || e.message.contains('user_type'))) {
        setState(() => _err =
            'Account type mismatch — please re-register selecting the Lawyer account type.');
      } else {
        setState(() => _err = e.message);
      }
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final user = context.watch<AuthProvider>().user;
    final isLawyerAccount = user?.isLawyerAccount ?? false;
    final appBarTitle = isLawyerAccount ? 'Lawyer Verification' : 'Apply as Lawyer';
    return Scaffold(
      appBar: LegatoAppBar(
        title: Text(appBarTitle),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadingStatus ? null : _loadStatus,
          ),
        ],
      ),
      body: _loadingStatus
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _StatusBanner(statusData: _statusData, cs: cs),
                  const SizedBox(height: 24),
                  if (_canReapply) ...[
                    Text(
                      'Submit your documents',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Upload your CV and national ID card. An admin will review and verify your lawyer credentials.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                    ),
                    const SizedBox(height: 16),
                    // CV picker
                    _buildDocPicker(
                      context,
                      label: 'CV / Resume',
                      icon: Icons.description_outlined,
                      filename: _cvFilename ?? (_hasServerCv ? (_serverCvFilename ?? 'Previously uploaded') : null),
                      isServerFile: _cvFilename == null && _hasServerCv,
                      onPick: _pickCv,
                      cs: cs,
                    ),
                    const SizedBox(height: 12),
                    // ID card picker
                    _buildDocPicker(
                      context,
                      label: 'National ID Card',
                      icon: Icons.badge_outlined,
                      filename: _idCardFilename ?? (_hasServerIdCard ? (_serverIdCardFilename ?? 'Previously uploaded') : null),
                      isServerFile: _idCardFilename == null && _hasServerIdCard,
                      onPick: _pickIdCard,
                      cs: cs,
                    ),
                    const SizedBox(height: 12),
                    // Optional bar license number
                    TextFormField(
                      controller: _licenseController,
                      decoration: const InputDecoration(
                        labelText: 'Bar License Number (optional)',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.numbers_outlined),
                      ),
                    ),
                    const SizedBox(height: 12),
                    TextFormField(
                      controller: _yearsController,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        labelText: 'Years of Experience (optional)',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.work_history_outlined),
                      ),
                    ),
                    if (_err != null) ...[
                      const SizedBox(height: 12),
                      Text(_err!, style: TextStyle(color: cs.error)),
                    ],
                    if (_successMsg != null) ...[
                      const SizedBox(height: 12),
                      Text(_successMsg!, style: TextStyle(color: cs.primary)),
                    ],
                    const SizedBox(height: 20),
                    FilledButton.icon(
                      onPressed: _submitting ? null : _submit,
                      icon: _submitting
                          ? const SizedBox(
                              height: 18,
                              width: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.send_outlined),
                      label: const Text('Submit Application'),
                    ),
                  ],
                ],
              ),
            ),
    );
  }

  Widget _buildDocPicker(
    BuildContext context, {
    required String label,
    required IconData icon,
    required String? filename,
    required VoidCallback onPick,
    required ColorScheme cs,
    bool isServerFile = false,
  }) {
    final picked = filename != null;
    final borderColor = picked ? (isServerFile ? Colors.green : cs.primary) : cs.outlineVariant;
    final fgColor = picked ? (isServerFile ? Colors.green : cs.primary) : cs.onSurfaceVariant;
    return OutlinedButton(
      style: OutlinedButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        side: BorderSide(color: borderColor),
        foregroundColor: fgColor,
        alignment: Alignment.centerLeft,
      ),
      onPressed: _submitting ? null : onPick,
      child: Row(
        children: [
          Icon(picked ? Icons.check_circle_outline : icon, size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label, style: Theme.of(context).textTheme.labelMedium),
                if (filename != null)
                  Text(
                    isServerFile ? 'On server: $filename (tap to replace)' : filename,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: isServerFile ? Colors.green : cs.primary,
                        ),
                    overflow: TextOverflow.ellipsis,
                  )
                else
                  Text(
                    'Tap to select file (PDF / image)',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
              ],
            ),
          ),
          Icon(Icons.upload_file_outlined, size: 18, color: cs.onSurfaceVariant),
        ],
      ),
    );
  }

  bool get _canReapply {
    final status = _statusData?['status'] as String? ?? 'not_applied';
    return status == 'not_applied' || status == 'rejected';
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({required this.statusData, required this.cs});

  final Map<String, dynamic>? statusData;
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    final status = statusData?['status'] as String? ?? 'not_applied';

    IconData icon;
    Color color;
    String title;
    String body;

    switch (status) {
      case 'approved':
        icon = Icons.verified_outlined;
        color = Colors.green;
        title = 'Verified Lawyer';
        body = 'Your lawyer status has been approved. All lawyer features are now active.';
      case 'pending':
        icon = Icons.hourglass_empty_outlined;
        color = cs.secondary;
        title = 'Under Review';
        body = 'Your application is being reviewed by an admin. You will be notified in the Alerts tab once a decision is made.';
      case 'rejected':
        icon = Icons.cancel_outlined;
        color = cs.error;
        title = 'Application Rejected';
        body = (statusData?['admin_note'] as String?)?.isNotEmpty == true
            ? 'Reason: ${statusData?['admin_note']}'
            : 'Your application was not approved. Please re-apply with updated documents.';
      default:
        icon = Icons.gavel_outlined;
        color = cs.primary;
        title = 'Apply as a Lawyer';
        body = 'Upload your CV / resume and national ID card to request lawyer status. '
            'An admin will review your documents and you will be notified in the Alerts tab.';
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: Theme.of(context)
                      .textTheme
                      .titleSmall
                      ?.copyWith(color: color, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4),
                Text(body, style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
