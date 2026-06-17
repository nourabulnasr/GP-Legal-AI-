import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import 'package:legato_mobile/config/app_config.dart';
import 'package:legato_mobile/providers/auth_provider.dart';
import 'package:legato_mobile/screens/admin/admin_screen.dart';
import 'package:legato_mobile/screens/features/features_hub_screen.dart';
import 'package:legato_mobile/screens/roadmap/roadmap_screen.dart';
import 'package:legato_mobile/screens/settings/settings_screen.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

class MoreScreen extends StatelessWidget {
  const MoreScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final user = context.watch<AuthProvider>().user;

    return LegatoPageScaffold(
      title: 'More',
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          children: [
            Text(
              'Tools and preferences',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: LegatoLinkedInTheme.textSecondaryAdaptive(context)),
            ),
            const SizedBox(height: 20),
            _SectionLabel(title: 'Legal workspace'),
            Card(
              child: ListTile(
                leading: Icon(Icons.apps_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: Text(FeaturesHubScreen.allToolsLabel),
                subtitle: Text('${FeaturesHubScreen.toolCount} Legato tools — compare, explain, share, and more'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const FeaturesHubScreen()),
                ),
              ),
            ),
            Card(
              child: ListTile(
                leading: Icon(Icons.flag_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: const Text('Roadmap'),
                subtitle: const Text('Phase notes and integrations'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const RoadmapScreen()),
                ),
              ),
            ),
            if (user?.isAdmin ?? false) ...[
              _SectionLabel(title: 'Administration'),
              Card(
                child: ListTile(
                  leading: Icon(Icons.admin_panel_settings_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                  title: const Text('Admin'),
                  subtitle: const Text('Users and analyses'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(builder: (_) => const AdminScreen()),
                  ),
                ),
              ),
            ],
            if (user?.isAdmin ?? false)
            Card(
              child: ListTile(
                leading: Icon(Icons.open_in_browser_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: const Text('Backend API & law admin'),
                subtitle: const Text('Opens Swagger /docs (admin-law routes; login in browser if needed)'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () async {
                  final u = AppConfig.backendDocsUri();
                  final messenger = ScaffoldMessenger.of(context);
                  if (await canLaunchUrl(u)) {
                    await launchUrl(u, mode: LaunchMode.externalApplication);
                  } else {
                    messenger.showSnackBar(
                      SnackBar(content: Text('Cannot open browser on this device. URL: $u')),
                    );
                  }
                },
              ),
            ),
            _SectionLabel(title: 'Account'),
            Card(
              child: ListTile(
                leading: Icon(Icons.settings_outlined, color: LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.95)),
                title: const Text('Settings'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const SettingsScreen()),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel({required this.title});

  final String title;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, bottom: 8, top: 8),
      child: Text(
        title,
        style: Theme.of(context).textTheme.labelLarge?.copyWith(
              fontWeight: FontWeight.w700,
              color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
            ),
      ),
    );
  }
}
