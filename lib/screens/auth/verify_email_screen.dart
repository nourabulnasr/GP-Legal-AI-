import 'package:flutter/material.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/services/auth_service.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:provider/provider.dart';

class VerifyEmailScreen extends StatefulWidget {
  const VerifyEmailScreen({super.key, required this.initialEmail});

  final String initialEmail;

  @override
  State<VerifyEmailScreen> createState() => _VerifyEmailScreenState();
}

class _VerifyEmailScreenState extends State<VerifyEmailScreen> {
  late final TextEditingController _email;
  final _code = TextEditingController();
  BuildContext? _formCtx;
  bool _busy = false;
  String? _err;
  String? _ok;

  @override
  void initState() {
    super.initState();
    _email = TextEditingController(text: widget.initialEmail);
  }

  @override
  void dispose() {
    _email.dispose();
    _code.dispose();
    super.dispose();
  }

  AuthService get _auth => context.read<AppServices>().auth;

  Future<void> _verify() async {
    final ok = _formCtx != null && (Form.of(_formCtx!).validate());
    if (!ok) return;
    setState(() {
      _busy = true;
      _err = null;
      _ok = null;
    });
    try {
      final r = await _auth.verifyEmail(email: _email.text, code: _code.text);
      setState(() => _ok = r['message']?.toString() ?? 'Verified.');
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _resend() async {
    setState(() {
      _busy = true;
      _err = null;
    });
    try {
      await _auth.resendVerification(_email.text);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('If eligible, a new code was sent.')),
        );
      }
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
      appBar: AppBar(title: const Text('Verify email')),
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
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _code,
                  decoration: const InputDecoration(
                    labelText: 'Code from email',
                    border: OutlineInputBorder(),
                  ),
                  validator: (v) =>
                      (v == null || v.trim().isEmpty) ? 'Enter code' : null,
                ),
                if (_err != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 12),
                    child: Text(
                      _err!,
                      style: TextStyle(color: Theme.of(context).colorScheme.error),
                    ),
                  ),
                if (_ok != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 12),
                    child: Text(_ok!, style: const TextStyle(color: Colors.green)),
                  ),
                const SizedBox(height: 20),
                FilledButton(
                  onPressed: _busy ? null : _verify,
                  child: const Text('Verify'),
                ),
                TextButton(
                  onPressed: _busy ? null : _resend,
                  child: const Text('Resend code'),
                ),
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('Back to sign in'),
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
