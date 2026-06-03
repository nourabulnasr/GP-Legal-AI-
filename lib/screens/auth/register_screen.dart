import 'package:flutter/material.dart';
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

  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final ok = _formCtx != null && (Form.of(_formCtx!).validate());
    if (!ok) return;
    setState(() {
      _busy = true;
      _err = null;
    });
    try {
      await context.read<AuthProvider>().register(_email.text, _password.text);
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
    return Scaffold(
      appBar: AppBar(title: const Text('Register')),
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
                if (_err != null) ...[
                  const SizedBox(height: 12),
                  Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
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
