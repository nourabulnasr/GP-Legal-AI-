import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/auth/login_screen.dart';
import 'package:legato_mobile/screens/home/home_shell.dart';

class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  bool _wasAuthenticated = false;
  bool _dialogPending = false;

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();

    if (auth.loading) {
      return const Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Legato'),
            ],
          ),
        ),
      );
    }

    final nowAuthenticated = auth.isAuthenticated;

    // Session expired: was logged in, now not, and an error message is present
    if (_wasAuthenticated && !nowAuthenticated && auth.error != null && !_dialogPending) {
      _dialogPending = true;
      final msg = auth.error!;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        _dialogPending = false;
        showDialog<void>(
          context: context,
          barrierDismissible: false,
          builder: (ctx) => AlertDialog(
            title: const Text('Session expired'),
            content: Text(msg),
            actions: [
              FilledButton(
                onPressed: () {
                  Navigator.of(ctx).pop();
                  context.read<AuthProvider>().clearError();
                },
                child: const Text('Sign in again'),
              ),
            ],
          ),
        );
      });
    }

    _wasAuthenticated = nowAuthenticated;

    if (nowAuthenticated) return const HomeShell();
    return const LoginScreen();
  }
}
