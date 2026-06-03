import 'package:flutter/material.dart';

import 'package:legato_mobile/screens/home/dashboard_tab.dart';
import 'package:legato_mobile/screens/social/alerts_screen.dart';
import 'package:legato_mobile/screens/social/contracts_tab_screen.dart';
import 'package:legato_mobile/screens/social/feed_screen.dart';
import 'package:legato_mobile/screens/social/network_screen.dart';
import 'package:legato_mobile/screens/social/profile_screen.dart';
/// Main shell: Feed | Network | Contracts | Alerts | Profile (LexConnect-style bottom nav).
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;
  final _alertsKey = GlobalKey<AlertsScreenState>();

  static const _tabs = [
    _TabSpec('Home', Icons.dashboard_outlined, Icons.dashboard),
    _TabSpec('Feed', Icons.article_outlined, Icons.article),
    _TabSpec('Network', Icons.people_outline, Icons.people),
    _TabSpec('Contracts', Icons.description_outlined, Icons.description),
    _TabSpec('Alerts', Icons.notifications_outlined, Icons.notifications),
    _TabSpec('Profile', Icons.person_outline, Icons.person),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _index,
        children: [
          const DashboardTab(),
          const FeedScreen(),
          const NetworkScreen(),
          const ContractsTabScreen(),
          AlertsScreen(key: _alertsKey),
          const ProfileScreen(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        surfaceTintColor: Colors.transparent,
        selectedIndex: _index,
        onDestinationSelected: (i) {
          if (i == 4) _alertsKey.currentState?.refresh();
          setState(() => _index = i);
        },
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        destinations: [
          for (final t in _tabs)
            NavigationDestination(
              icon: Icon(t.outlined),
              selectedIcon: Icon(t.filled),
              label: t.label,
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
