import 'package:flutter/material.dart';

import 'package:legato_mobile/theme/linkedin_theme.dart';

/// Circular avatar with network image and initials fallback.
class UserAvatar extends StatelessWidget {
  const UserAvatar({
    super.key,
    this.imageUrl,
    required this.name,
    this.radius = 22,
    this.backgroundColor,
  });

  final String? imageUrl;
  final String name;
  final double radius;
  final Color? backgroundColor;

  String get _initial {
    final trimmed = name.trim();
    if (trimmed.isEmpty) return '?';
    return trimmed[0].toUpperCase();
  }

  @override
  Widget build(BuildContext context) {
    final url = imageUrl?.trim();
    final fallbackBg = backgroundColor ?? LegatoLinkedInTheme.navActiveGold.withValues(alpha: 0.14);
    final size = radius * 2;

    if (url == null || url.isEmpty) {
      return CircleAvatar(
        radius: radius,
        backgroundColor: fallbackBg,
        child: Text(
          _initial,
          style: TextStyle(
            fontWeight: FontWeight.w800,
            fontSize: radius * 0.85,
            color: const Color(0xFF8B7318),
          ),
        ),
      );
    }

    return CircleAvatar(
      radius: radius,
      backgroundColor: fallbackBg,
      child: ClipOval(
        child: Image.network(
          url,
          key: ValueKey(url),
          width: size,
          height: size,
          fit: BoxFit.cover,
          errorBuilder: (_, err, stack) => _fallback(fallbackBg),
          loadingBuilder: (context, child, progress) {
            if (progress == null) return child;
            return _fallback(fallbackBg);
          },
        ),
      ),
    );
  }

  Widget _fallback(Color bg) {
    return ColoredBox(
      color: bg,
      child: Center(
        child: Text(
          _initial,
          style: TextStyle(
            fontWeight: FontWeight.w800,
            fontSize: radius * 0.85,
            color: const Color(0xFF8B7318),
          ),
        ),
      ),
    );
  }
}
