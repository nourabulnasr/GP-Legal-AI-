import 'package:flutter/material.dart';

import 'package:legato_mobile/widgets/legato_app_bar.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';

/// POST /auth/forgot-password → /auth/verify-reset-code → /auth/reset-password
class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _email = TextEditingController();
  final _code = TextEditingController();
  final _password = TextEditingController();
  BuildContext? _formCtx;
  int _step = 0;
  bool _busy = false;
  String? _err;

  @override
  void dispose() {
    _email.dispose();
    _code.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _submitEmail() async {
    final ok = _formCtx != null && (Form.of(_formCtx!).validate());
    if (!ok) return;
    setState(() {
      _busy = true;
      _err = null;
    });
    try {
      await context.read<AppServices>().auth.forgotPassword(_email.text);
      if (!mounted) return;
      setState(() => _step = 1);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('If an account exists, a reset code was sent to your email.')),
      );
    } on ApiException catch (e) {
      setState(() => _err = e.message);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _submitCodeAndPassword() async {
    final ok = _formCtx != null && (Form.of(_formCtx!).validate());
    if (!ok) return;
    setState(() {
      _busy = true;
      _err = null;
    });
    final auth = context.read<AppServices>().auth;
    try {
      final verify = await auth.verifyResetCode(
        email: _email.text,
        code: _code.text,
      );
      final token = verify['reset_token'] as String?;
      if (token == null || token.isEmpty) {
        throw ApiException('No reset token returned');
      }
      await auth.resetPassword(resetToken: token, newPassword: _password.text);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Password updated. You can sign in now.')),
      );
      Navigator.of(context).pop(true);
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
      appBar: LegatoAppBar(title: const Text('Reset password')),
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
                if (_step == 0) ...[
                  Text(
                    'Enter your account email. We will send a reset code if the account exists.',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 16),
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
                ] else ...[
                  Text(
                    'Enter the code from your email and choose a new password (min. 6 characters).',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 16),
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
                    controller: _code,
                    decoration: const InputDecoration(
                      labelText: 'Reset code',
                      border: OutlineInputBorder(),
                    ),
                    validator: (v) =>
                        (v == null || v.trim().isEmpty) ? 'Enter code' : null,
                  ),
                  const SizedBox(height: 12),
                  TextFormField(
                    controller: _password,
                    obscureText: true,
                    decoration: const InputDecoration(
                      labelText: 'New password',
                      border: OutlineInputBorder(),
                    ),
                    validator: (v) =>
                        (v == null || v.length < 6) ? 'Min 6 characters' : null,
                  ),
                ],
                if (_err != null) ...[
                  const SizedBox(height: 12),
                  Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                ],
                const SizedBox(height: 20),
                FilledButton(
                  onPressed: _busy
                      ? null
                      : (_step == 0 ? _submitEmail : _submitCodeAndPassword),
                  child: _busy
                      ? const SizedBox(
                          height: 22,
                          width: 22,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : Text(_step == 0 ? 'Send code' : 'Update password'),
                ),
                if (_step == 1)
                  TextButton(
                    onPressed: _busy ? null : () => setState(() => _step = 0),
                    child: const Text('Back'),
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
