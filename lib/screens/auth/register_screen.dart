import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/config/runtime_config.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/auth/verify_email_screen.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _email = TextEditingController();
  final _password = TextEditingController();
  BuildContext? _formCtx;
  bool _busy = false;
  String? _err;
  String _userType = 'user';

  Uint8List? _cvBytes;
  String? _cvFilename;
  Uint8List? _idCardFrontBytes;
  String? _idCardFrontFilename;
  Uint8List? _idCardBackBytes;
  String? _idCardBackFilename;
  final _years = TextEditingController();
  final _hourlyRate = TextEditingController();

  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    _years.dispose();
    _hourlyRate.dispose();
    super.dispose();
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

  Future<void> _pickIdFromFile({required bool isFront}) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['jpg', 'jpeg', 'png', 'pdf'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;
    final file = result.files.first;
    if (file.bytes == null) return;
    setState(() {
      if (isFront) {
        _idCardFrontBytes = file.bytes;
        _idCardFrontFilename = file.name;
      } else {
        _idCardBackBytes = file.bytes;
        _idCardBackFilename = file.name;
      }
    });
  }

  Future<void> _pickIdFromCamera({required bool isFront}) async {
    final picker = ImagePicker();
    final photo = await picker.pickImage(source: ImageSource.camera, imageQuality: 85);
    if (photo == null) return;
    final bytes = await photo.readAsBytes();
    if (!mounted) return;
    setState(() {
      final name = photo.name.isNotEmpty ? photo.name : 'id_${isFront ? 'front' : 'back'}.jpg';
      if (isFront) {
        _idCardFrontBytes = bytes;
        _idCardFrontFilename = name;
      } else {
        _idCardBackBytes = bytes;
        _idCardBackFilename = name;
      }
    });
  }

  Future<void> _pickIdCardFront() => _pickIdFromFile(isFront: true);

  Future<void> _pickIdCardBack() => _pickIdFromFile(isFront: false);

  Future<void> _signInWithGoogle() async {
    setState(() { _busy = true; _err = null; });
    try {
      if (kIsWeb) {
        final err = await context.read<AuthProvider>()
            .signInWithGoogleWeb(userType: _userType);
        if (err != null && mounted) setState(() => _err = err);
      } else {
        await context.read<AuthProvider>()
            .signInWithGoogleNative(userType: _userType);
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _err = e.message);
    } catch (e) {
      if (mounted) setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _submit() async {
    final ok = _formCtx != null && (Form.of(_formCtx!).validate());
    if (!ok) return;

    if (_userType == 'lawyer' &&
        (_cvBytes == null || _idCardFrontBytes == null || _idCardBackBytes == null)) {
      setState(() => _err = 'Please upload your CV and both sides of your ID card to register as a lawyer.');
      return;
    }
    if (_userType == 'lawyer') {
      final years = int.tryParse(_years.text.trim());
      final rate = double.tryParse(_hourlyRate.text.trim());
      if (years == null || years < 0) {
        setState(() => _err = 'Please enter your years of experience.');
        return;
      }
      if (rate == null || rate <= 0) {
        setState(() => _err = 'Please enter your hourly rate.');
        return;
      }
    }

    setState(() {
      _busy = true;
      _err = null;
    });
    try {
      await context.read<AuthProvider>().register(
            _email.text,
            _password.text,
            userType: _userType,
            cvBytes: _cvBytes,
            cvFilename: _cvFilename,
            idCardBytes: _idCardFrontBytes,
            idCardFilename: _idCardFrontFilename,
            idCardBackBytes: _idCardBackBytes,
            idCardBackFilename: _idCardBackFilename,
            yearsOfExperience: int.tryParse(_years.text.trim()),
            hourlyRate: double.tryParse(_hourlyRate.text.trim()),
          );
      if (!mounted) return;
      await Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(
          builder: (_) => VerifyEmailScreen(initialEmail: _email.text.trim()),
        ),
      );
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } catch (e) {
      final msg = e.toString();
      if (msg.contains('Failed to fetch') && msg.contains('76.13.4.148')) {
        setState(() => _err =
            'Cannot reach the API over HTTP from this app. Clear site data (web) or reinstall the app, then use https://srv1723974.hstgr.cloud');
      } else if (msg.contains('Failed to fetch')) {
        setState(() => _err =
            'Network error reaching the API (${RuntimeConfig.apiBaseUrl}). Check your connection or Settings → API URL.');
      } else {
        setState(() => _err = msg);
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: LegatoAppBar(title: const Text('Register')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Form(
            child: Builder(
              builder: (formCtx) {
                _formCtx = formCtx;
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    TextFormField(
                      controller: _email,
                      keyboardType: TextInputType.emailAddress,
                      decoration: const InputDecoration(
                        labelText: 'Email',
                        border: OutlineInputBorder(),
                      ),
                      validator: (v) =>
                          (v == null || v.trim().isEmpty) ? 'Enter email' : null,
                    ),
                    const SizedBox(height: 12),
                    TextFormField(
                      controller: _password,
                      obscureText: true,
                      decoration: const InputDecoration(
                        labelText: 'Password (min 6)',
                        border: OutlineInputBorder(),
                      ),
                      validator: (v) =>
                          (v == null || v.length < 6) ? 'Min 6 characters' : null,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Account type',
                      style: Theme.of(context).textTheme.labelLarge,
                    ),
                    const SizedBox(height: 8),
                    _AccountTypeCard(
                      selected: _userType == 'user',
                      icon: Icons.person_outline,
                      title: 'Regular User',
                      subtitle: 'Analyze contracts and use all legal tools.',
                      onTap: () => setState(() => _userType = 'user'),
                    ),
                    const SizedBox(height: 8),
                    _AccountTypeCard(
                      selected: _userType == 'lawyer',
                      icon: Icons.gavel_outlined,
                      title: 'Lawyer',
                      subtitle:
                          'Register as a verified lawyer. Upload your CV and both sides of your ID card for admin approval.',
                      onTap: () => setState(() => _userType = 'lawyer'),
                    ),
                    if (_userType == 'lawyer') ...[
                      const SizedBox(height: 16),
                      Text(
                        'Lawyer documents',
                        style: Theme.of(context).textTheme.labelLarge,
                      ),
                      const SizedBox(height: 8),
                      _DocPickerTile(
                        label: 'CV / Resume',
                        icon: Icons.description_outlined,
                        filename: _cvFilename,
                        onPick: _pickCv,
                        required: true,
                        hint: 'PDF / doc / image',
                      ),
                      const SizedBox(height: 8),
                      _DocPickerTile(
                        label: 'National ID Card (Front)',
                        icon: Icons.badge_outlined,
                        filename: _idCardFrontFilename,
                        onPick: _pickIdCardFront,
                        onCamera: kIsWeb ? null : () => _pickIdFromCamera(isFront: true),
                        required: true,
                        hint: 'photo or file',
                      ),
                      const SizedBox(height: 8),
                      _DocPickerTile(
                        label: 'National ID Card (Back)',
                        icon: Icons.badge_outlined,
                        filename: _idCardBackFilename,
                        onPick: _pickIdCardBack,
                        onCamera: kIsWeb ? null : () => _pickIdFromCamera(isFront: false),
                        required: true,
                        hint: 'photo or file',
                      ),
                      const SizedBox(height: 8),
                      TextFormField(
                        controller: _years,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          labelText: 'Years of Experience *',
                          border: OutlineInputBorder(),
                        ),
                        validator: (v) {
                          if (_userType != 'lawyer') return null;
                          final n = int.tryParse((v ?? '').trim());
                          if (n == null || n < 0) return 'Enter years of experience';
                          return null;
                        },
                      ),
                      const SizedBox(height: 8),
                      TextFormField(
                        controller: _hourlyRate,
                        keyboardType: const TextInputType.numberWithOptions(decimal: true),
                        decoration: const InputDecoration(
                          labelText: 'Pricing per hour *',
                          border: OutlineInputBorder(),
                          prefixIcon: Icon(Icons.payments_outlined),
                          hintText: 'e.g. 500',
                        ),
                        validator: (v) {
                          if (_userType != 'lawyer') return null;
                          final n = double.tryParse((v ?? '').trim());
                          if (n == null || n <= 0) return 'Enter hourly rate';
                          return null;
                        },
                      ),
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: cs.secondaryContainer.withValues(alpha: 0.4),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: cs.secondary.withValues(alpha: 0.4)),
                        ),
                        child: Row(
                          children: [
                            Icon(Icons.info_outline, size: 18, color: cs.secondary),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Your documents will be reviewed by an admin. You will be notified once your account is approved.',
                                style: Theme.of(context)
                                    .textTheme
                                    .bodySmall
                                    ?.copyWith(color: cs.onSecondaryContainer),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                    if (_err != null) ...[
                      const SizedBox(height: 12),
                      Text(_err!, style: TextStyle(color: cs.error)),
                    ],
                    const SizedBox(height: 20),
                    FilledButton(
                      onPressed: _busy ? null : _submit,
                      child: _busy
                          ? const SizedBox(
                              height: 22,
                              width: 22,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Text('Register'),
                    ),
                    // Google OAuth cannot carry file uploads, so hide it for
                    // the Lawyer account type.
                    if (_userType != 'lawyer') ...[
                      const SizedBox(height: 8),
                      OutlinedButton(
                        onPressed: _busy ? null : _signInWithGoogle,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              'G',
                              style: TextStyle(
                                color: _busy ? Colors.grey : const Color(0xFF4285F4),
                                fontWeight: FontWeight.w700,
                                fontSize: 18,
                              ),
                            ),
                            const SizedBox(width: 10),
                            const Text('Continue with Google'),
                          ],
                        ),
                      ),
                    ],
                    TextButton(
                      onPressed: _busy ? null : () => Navigator.of(context).pop(),
                      child: const Text('Already have an account? Sign in'),
                    ),
                  ],
                );
              },
            ),
          ),
        ),
      ),
    );
  }
}

class _DocPickerTile extends StatelessWidget {
  const _DocPickerTile({
    required this.label,
    required this.icon,
    required this.filename,
    required this.onPick,
    this.required = false,
    this.hint = 'file',
    this.onCamera,
  });

  final String label;
  final IconData icon;
  final String? filename;
  final VoidCallback onPick;
  final bool required;
  final String hint;
  final VoidCallback? onCamera;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final picked = filename != null;
    final borderColor = picked ? cs.primary : (required ? cs.error.withValues(alpha: 0.5) : cs.outlineVariant);
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onPick,
        borderRadius: BorderRadius.circular(4),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: borderColor),
          ),
          child: Row(
            children: [
              Icon(picked ? Icons.check_circle_outline : icon, size: 20, color: picked ? cs.primary : cs.onSurfaceVariant),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  picked ? filename! : '$label${required ? ' *' : ''} ($hint)',
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(color: picked ? cs.primary : cs.onSurfaceVariant),
                ),
              ),
              if (onCamera != null)
                IconButton(
                  tooltip: 'Take a photo',
                  icon: Icon(Icons.photo_camera_outlined, size: 22, color: cs.primary),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
                  onPressed: onCamera,
                ),
              Icon(Icons.upload_file_outlined, size: 16, color: cs.onSurfaceVariant),
            ],
          ),
        ),
      ),
    );
  }
}

class _AccountTypeCard extends StatelessWidget {
  const _AccountTypeCard({
    required this.selected,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final bool selected;
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(10),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: selected ? cs.primary : cs.outlineVariant,
            width: selected ? 2 : 1,
          ),
          color: selected ? cs.primaryContainer.withValues(alpha: 0.35) : Colors.transparent,
        ),
        child: Row(
          children: [
            Icon(icon, color: selected ? cs.primary : cs.onSurfaceVariant),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          color: selected ? cs.primary : null,
                          fontWeight: FontWeight.w600,
                        ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: cs.onSurfaceVariant),
                  ),
                ],
              ),
            ),
            if (selected)
              Icon(Icons.check_circle, color: cs.primary, size: 20),
          ],
        ),
      ),
    );
  }
}
