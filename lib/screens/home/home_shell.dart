import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/screens/home/dashboard_tab.dart';
import 'package:legato_mobile/screens/messaging/messages_hub_screen.dart';
import 'package:legato_mobile/screens/social/alerts_screen.dart';
import 'package:legato_mobile/screens/social/feed_screen.dart';
import 'package:legato_mobile/screens/social/network_screen.dart';
import 'package:legato_mobile/screens/social/profile_screen.dart';

/// Main shell: Home | Feed | Network | Messages | Alerts | Profile.
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;
  int _alertsUnread = 0;
  Timer? _badgeTimer;
  final _homeKey = GlobalKey<DashboardTabState>();
  final _feedKey = GlobalKey<FeedScreenState>();
  final _networkKey = GlobalKey<NetworkScreenState>();
  final _messagesKey = GlobalKey<MessagesHubScreenState>();
  final _alertsKey = GlobalKey<AlertsScreenState>();
  final _profileKey = GlobalKey<ProfileScreenState>();

  static const _tabs = [
    _TabSpec('Home', Icons.dashboard_outlined, Icons.dashboard),
    _TabSpec('Feed', Icons.article_outlined, Icons.article),
    _TabSpec('Network', Icons.people_outline, Icons.people),
    _TabSpec('Messages', Icons.chat_bubble_outline, Icons.chat_bubble),
    _TabSpec('Alerts', Icons.notifications_outlined, Icons.notifications),
    _TabSpec('Profile', Icons.person_outline, Icons.person),
  ];

  static const _alertsTabIndex = 4;

  @override
  void initState() {
    super.initState();
    _refreshBadge();
    _badgeTimer = Timer.periodic(const Duration(seconds: 30), (_) => _refreshBadge());
  }

  @override
  void dispose() {
    _badgeTimer?.cancel();
    super.dispose();
  }

  Future<void> _refreshBadge() async {
    try {
      final n = await context.read<AppServices>().legato.unreadNotificationCount();
      if (mounted) setState(() => _alertsUnread = n);
    } catch (_) {}
  }

  Future<void> _openPostFromAlert(int postId) async {
    setState(() => _index = 1);
    // Wait for IndexedStack to show Feed before scrolling to the post.
    await Future<void>.delayed(Duration.zero);
    if (!mounted) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      WidgetsBinding.instance.addPostFrameCallback((_) async {
        if (!mounted) return;
        await _feedKey.currentState?.openPostById(postId);
      });
    });
  }

  void _refreshTab(int index) {
    switch (index) {
      case 0:
        _homeKey.currentState?.refresh();
      case 1:
        _feedKey.currentState?.refresh();
      case 2:
        _networkKey.currentState?.refresh();
      case 3:
        _messagesKey.currentState?.refresh();
      case 4:
        _alertsKey.currentState?.refresh();
      case 5:
        _profileKey.currentState?.refresh();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _index,
        children: [
          DashboardTab(key: _homeKey),
          FeedScreen(key: _feedKey),
          NetworkScreen(key: _networkKey),
          MessagesHubScreen(key: _messagesKey),
          AlertsScreen(key: _alertsKey, onOpenPost: _openPostFromAlert),
          ProfileScreen(key: _profileKey),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        surfaceTintColor: Colors.transparent,
        selectedIndex: _index,
        onDestinationSelected: (i) {
          setState(() => _index = i);
          _refreshTab(i);
          if (i == _alertsTabIndex) _refreshBadge();
        },
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        destinations: [
          for (var i = 0; i < _tabs.length; i++)
            NavigationDestination(
              icon: i == _alertsTabIndex && _alertsUnread > 0
                  ? Badge(
                      label: Text(_alertsUnread > 99 ? '99+' : '$_alertsUnread'),
                      child: Icon(_tabs[i].outlined),
                    )
                  : Icon(_tabs[i].outlined),
              selectedIcon: i == _alertsTabIndex && _alertsUnread > 0
                  ? Badge(
                      label: Text(_alertsUnread > 99 ? '99+' : '$_alertsUnread'),
                      child: Icon(_tabs[i].filled),
                    )
                  : Icon(_tabs[i].filled),
              label: _tabs[i].label,
            ),
        ],
      ),
    );
  }
}

class _TabSpec {
  const _TabSpec(this.label, this.outlined, this.filled);
  final String label;
  final IconData outlined;
  final IconData filled;
}
