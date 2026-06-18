import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/providers/user_profile_provider.dart';
import 'package:legato_mobile/screens/home/dashboard_tab.dart';
import 'package:legato_mobile/screens/messaging/messages_hub_screen.dart';
import 'package:legato_mobile/screens/social/alerts_screen.dart';
import 'package:legato_mobile/screens/social/feed_screen.dart';
import 'package:legato_mobile/screens/social/network_screen.dart';
import 'package:legato_mobile/screens/social/profile_screen.dart';

/// Main shell: Home | Feed | Network | Messages | Alerts | Profile.
/// Verified lawyers see Feed | Messages | Alerts | Profile only.
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;
  int _alertsUnread = 0;
  Timer? _badgeTimer;
  AuthProvider? _authProvider;
  bool? _lastAuthState;
  bool? _lastLawyerMode;
  final _homeKey = GlobalKey<DashboardTabState>();
  final _feedKey = GlobalKey<FeedScreenState>();
  final _networkKey = GlobalKey<NetworkScreenState>();
  final _messagesKey = GlobalKey<MessagesHubScreenState>();
  final _alertsKey = GlobalKey<AlertsScreenState>();
  final _profileKey = GlobalKey<ProfileScreenState>();

  static const _allTabs = [
    _TabSpec('Home', Icons.dashboard_outlined, Icons.dashboard, _ShellTab.home),
    _TabSpec('Feed', Icons.article_outlined, Icons.article, _ShellTab.feed),
    _TabSpec('Network', Icons.people_outline, Icons.people, _ShellTab.network),
    _TabSpec('Messages', Icons.chat_bubble_outline, Icons.chat_bubble, _ShellTab.messages),
    _TabSpec('Alerts', Icons.notifications_outlined, Icons.notifications, _ShellTab.alerts),
    _TabSpec('Profile', Icons.person_outline, Icons.person, _ShellTab.profile),
  ];

  static const _lawyerTabs = [
    _TabSpec('Feed', Icons.article_outlined, Icons.article, _ShellTab.feed),
    _TabSpec('Messages', Icons.chat_bubble_outline, Icons.chat_bubble, _ShellTab.messages),
    _TabSpec('Alerts', Icons.notifications_outlined, Icons.notifications, _ShellTab.alerts),
    _TabSpec('Profile', Icons.person_outline, Icons.person, _ShellTab.profile),
  ];

  List<_TabSpec> _tabsFor(bool lawyerMode) => lawyerMode ? _lawyerTabs : _allTabs;

  int _alertsIndexFor(bool lawyerMode) =>
      _tabsFor(lawyerMode).indexWhere((t) => t.id == _ShellTab.alerts);

  @override
  void initState() {
    super.initState();
    _refreshBadge();
    _badgeTimer = Timer.periodic(const Duration(seconds: 30), (_) => _refreshBadge());
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadUserProfile();
      _authProvider = context.read<AuthProvider>();
      _lastAuthState = _authProvider!.isAuthenticated;
      _lastLawyerMode = _authProvider!.user?.isVerifiedLawyer ?? false;
      _authProvider!.addListener(_onAuthStateChanged);
    });
  }

  Future<void> _loadUserProfile() async {
    final auth = context.read<AuthProvider>();
    await context.read<UserProfileProvider>().refresh(
          userId: auth.user?.id,
          email: auth.user?.email ?? '',
        );
  }

  void _onAuthStateChanged() {
    if (!mounted) return;
    final isAuth = _authProvider?.isAuthenticated ?? false;
    final lawyerMode = _authProvider?.user?.isVerifiedLawyer ?? false;
    if (_lastAuthState == isAuth && _lastLawyerMode == lawyerMode) return;
    _lastAuthState = isAuth;
    _lastLawyerMode = lawyerMode;
    if (lawyerMode && _index >= _lawyerTabs.length) {
      _index = 0;
    }
    setState(() {});
    if (isAuth) _loadUserProfile();
  }

  @override
  void dispose() {
    _badgeTimer?.cancel();
    _authProvider?.removeListener(_onAuthStateChanged);
    super.dispose();
  }

  Future<void> _refreshBadge() async {
    try {
      final n = await context.read<AppServices>().legato.unreadNotificationCount();
      if (mounted) setState(() => _alertsUnread = n);
    } catch (_) {}
  }

  Future<void> _openPostFromAlert(int postId) async {
    final tabs = _tabsFor(context.read<AuthProvider>().user?.isVerifiedLawyer ?? false);
    final feedIdx = tabs.indexWhere((t) => t.id == _ShellTab.feed);
    if (feedIdx < 0) return;
    setState(() => _index = feedIdx);
    await Future<void>.delayed(Duration.zero);
    if (!mounted) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      WidgetsBinding.instance.addPostFrameCallback((_) async {
        if (!mounted) return;
        await _feedKey.currentState?.openPostById(postId);
      });
    });
  }

  void _refreshTab(_ShellTab id) {
    switch (id) {
      case _ShellTab.home:
        _homeKey.currentState?.refresh();
      case _ShellTab.feed:
        _feedKey.currentState?.refresh();
      case _ShellTab.network:
        _networkKey.currentState?.refresh();
      case _ShellTab.messages:
        _messagesKey.currentState?.refresh();
      case _ShellTab.alerts:
        _alertsKey.currentState?.refresh();
      case _ShellTab.profile:
        _profileKey.currentState?.refresh();
    }
  }

  Widget _screenFor(_ShellTab id) {
    switch (id) {
      case _ShellTab.home:
        return DashboardTab(key: _homeKey);
      case _ShellTab.feed:
        return FeedScreen(key: _feedKey);
      case _ShellTab.network:
        return NetworkScreen(key: _networkKey);
      case _ShellTab.messages:
        return MessagesHubScreen(key: _messagesKey);
      case _ShellTab.alerts:
        return AlertsScreen(key: _alertsKey, onOpenPost: _openPostFromAlert);
      case _ShellTab.profile:
        return ProfileScreen(key: _profileKey);
    }
  }

  @override
  Widget build(BuildContext context) {
    final lawyerMode = context.watch<AuthProvider>().user?.isVerifiedLawyer ?? false;
    final tabs = _tabsFor(lawyerMode);
    final alertsIdx = _alertsIndexFor(lawyerMode);

    return Scaffold(
      body: IndexedStack(
        index: _index.clamp(0, tabs.length - 1),
        children: [for (final t in tabs) _screenFor(t.id)],
      ),
      bottomNavigationBar: NavigationBar(
        surfaceTintColor: Colors.transparent,
        selectedIndex: _index.clamp(0, tabs.length - 1),
        onDestinationSelected: (i) {
          setState(() => _index = i);
          _refreshTab(tabs[i].id);
          if (i == alertsIdx) _refreshBadge();
        },
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        destinations: [
          for (var i = 0; i < tabs.length; i++)
            NavigationDestination(
              icon: i == alertsIdx && _alertsUnread > 0
                  ? Badge(
                      label: Text(_alertsUnread > 99 ? '99+' : '$_alertsUnread'),
                      child: Icon(tabs[i].outlined),
                    )
                  : Icon(tabs[i].outlined),
              selectedIcon: i == alertsIdx && _alertsUnread > 0
                  ? Badge(
                      label: Text(_alertsUnread > 99 ? '99+' : '$_alertsUnread'),
                      child: Icon(tabs[i].filled),
                    )
                  : Icon(tabs[i].filled),
              label: tabs[i].label,
            ),
        ],
      ),
    );
  }
}

enum _ShellTab { home, feed, network, messages, alerts, profile }

class _TabSpec {
  const _TabSpec(this.label, this.outlined, this.filled, this.id);
  final String label;
  final IconData outlined;
  final IconData filled;
  final _ShellTab id;
}
