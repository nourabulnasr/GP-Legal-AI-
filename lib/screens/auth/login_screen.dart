import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/auth/forgot_password_screen.dart';
import 'package:legato_mobile/screens/auth/register_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
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
      context.read<AuthProvider>().clearError();
      await context.read<AuthProvider>().login(_email.text, _password.text);
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
    final sessionMsg = context.watch<AuthProvider>().error;

    return Scaffold(
      backgroundColor: LegatoLinkedInTheme.background,
      appBar: AppBar(title: const Text('Sign in')),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 400),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Form(
                    child: Builder(
                      builder: (formCtx) {
                        _formCtx = formCtx;
                        return Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                        Text(
                          'Legato',
                          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                                color: LegatoLinkedInTheme.navActiveGold,
                                fontWeight: FontWeight.w700,
                              ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Legal intelligence for your contracts.',
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                color: LegatoLinkedInTheme.textSecondary,
                              ),
                        ),
                if (sessionMsg != null && sessionMsg.isNotEmpty) ...[
                  const SizedBox(height: 16),
                  Material(
                    color: Theme.of(context).colorScheme.errorContainer,
                    borderRadius: BorderRadius.circular(8),
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Text(sessionMsg),
                    ),
                  ),
                ],
                const SizedBox(height: 24),
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
                    labelText: 'Password',
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
                      : const Text('Sign in'),
                ),
                TextButton(
                  onPressed: _busy
                      ? null
                      : () => Navigator.of(context).push(
                            MaterialPageRoute<void>(
                              builder: (_) => const RegisterScreen(),
                            ),
                          ),
                  child: const Text('Create account'),
                ),
                TextButton(
                  onPressed: _busy
                      ? null
                      : () async {
                          context.read<AuthProvider>().clearError();
                          await Navigator.of(context).push<bool>(
                            MaterialPageRoute<bool>(
                              builder: (_) => const ForgotPasswordScreen(),
                            ),
                          );
                        },
                  child: const Text('Forgot password?'),
                ),
                          ],
                        );
                      },
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
