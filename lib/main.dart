import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/config/runtime_config.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/auth_gate.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RuntimeConfig.init();
  FlutterError.onError = FlutterError.presentError;
  PlatformDispatcher.instance.onError = (error, stack) {
    FlutterError.reportError(FlutterErrorDetails(exception: error, stack: stack));
    return true;
  };
  runApp(const LegatoApp());
}

class LegatoApp extends StatelessWidget {
  const LegatoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider<AppServices>(create: (_) => AppServices()),
        ChangeNotifierProvider<AuthProvider>(
          create: (context) {
            final services = context.read<AppServices>();
            final p = AuthProvider(authService: services.auth);
            services.setOnUnauthorized(p.onSessionExpiredFromApi);
            p.bootstrap();
            return p;
          },
        ),
      ],
      child: _AppLifecycle(
        child: MaterialApp(
          title: 'Legato',
          theme: LegatoLinkedInTheme.light(),
          home: const AuthGate(),
        ),
      ),
    );
  }
}

/// Re-validates the session when the app returns to foreground (JWT expiry sync).
class _AppLifecycle extends StatefulWidget {
  const _AppLifecycle({required this.child});

  final Widget child;

  @override
  State<_AppLifecycle> createState() => _AppLifecycleState();
}

class _AppLifecycleState extends State<_AppLifecycle> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed && mounted) {
      context.read<AuthProvider>().refreshUser();
    }
  }

  @override
  Widget build(BuildContext context) => widget.child;
}
