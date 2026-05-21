import 'dart:async';
import 'dart:io';
import 'dart:ui';

import 'package:app_links/app_links.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/config/runtime_config.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/providers/theme_notifier.dart';
import 'package:legato_mobile/screens/auth_gate.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RuntimeConfig.init();
  final themeNotifier = await ThemeNotifier.init();
  FlutterError.onError = FlutterError.presentError;
  PlatformDispatcher.instance.onError = (error, stack) {
    FlutterError.reportError(FlutterErrorDetails(exception: error, stack: stack));
    return true;
  };
  runApp(LegatoApp(themeNotifier: themeNotifier));
}

class LegatoApp extends StatelessWidget {
  const LegatoApp({super.key, required this.themeNotifier});

  final ThemeNotifier themeNotifier;

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
        ChangeNotifierProvider<ThemeNotifier>.value(value: themeNotifier),
      ],
      child: _AppLifecycle(
        child: Consumer<ThemeNotifier>(
          builder: (_, theme, _) => MaterialApp(
            title: 'Legato',
            theme: LegatoLinkedInTheme.light(),
            darkTheme: LegatoLinkedInTheme.dark(),
            themeMode: theme.isDark ? ThemeMode.dark : ThemeMode.light,
            builder: (context, child) => ColoredBox(
              color: Theme.of(context).scaffoldBackgroundColor,
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 600),
                  child: child!,
                ),
              ),
            ),
            home: const AuthGate(),
          ),
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
  StreamSubscription<Uri>? _linkSub;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    if (!kIsWeb && Platform.isAndroid) {
      _initDeepLinks();
    }
  }

  Future<void> _initDeepLinks() async {
    final appLinks = AppLinks();
    try {
      final initialUri = await appLinks.getInitialLink();
      if (initialUri != null) {
        WidgetsBinding.instance.addPostFrameCallback((_) => _handleUri(initialUri));
      }
    } catch (e) {
      debugPrint('[DeepLink] getInitialLink error: $e');
    }
    _linkSub = appLinks.uriLinkStream.listen(_handleUri, onError: (e) {
      debugPrint('[DeepLink] stream error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Sign-in link error. Please try again.')),
        );
      }
    });
  }

  void _handleUri(Uri uri) {
    if (!mounted) return;
    final token = uri.queryParameters['token'];
    final error = uri.queryParameters['error'];
    if (token != null && token.isNotEmpty) {
      context.read<AuthProvider>().loginWithToken(token);
    } else if (error != null && error.isNotEmpty) {
      debugPrint('[DeepLink] OAuth error param: $error');
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Sign-in failed. Please try again.')),
      );
    }
  }

  @override
  void dispose() {
    _linkSub?.cancel();
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
