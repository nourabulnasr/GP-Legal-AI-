import 'package:legato_mobile/config/app_config.dart';

/// Extract a public share token from app or web URIs.
String? parseShareTokenFromUri(Uri uri) {
  final fromQuery = uri.queryParameters['share']?.trim();
  if (fromQuery != null && fromQuery.isNotEmpty) return fromQuery;

  final segments = uri.pathSegments.where((s) => s.isNotEmpty).toList();
  if (segments.length >= 2 && segments[0] == 'share') {
    return segments[1].trim().isEmpty ? null : segments[1].trim();
  }
  if (segments.length >= 4 &&
      segments[0] == 'legato' &&
      segments[1] == 'shares' &&
      segments[2] == 'public') {
    final token = segments[3].trim();
    return token.isEmpty ? null : token;
  }
  return null;
}

/// Web-friendly link: `https://legatoappgp2026.web.app/?share=TOKEN`
String buildPublicShareLink(String token) {
  final base = AppConfig.shareBaseUrl;
  return '$base/?share=${Uri.encodeComponent(token)}';
}

/// Clipboard-friendly text: readable label + URL on the next line.
String buildPublicShareMessage(String token) {
  return 'Shared Legato contract analysis (read-only)\n${buildPublicShareLink(token)}';
}

/// Pull a share token from plain text or a URL embedded in a chat message.
String? extractShareTokenFromText(String text) {
  final trimmed = text.trim();
  if (trimmed.isEmpty) return null;

  final directUri = Uri.tryParse(trimmed);
  if (directUri != null && directUri.hasScheme) {
    final token = parseShareTokenFromUri(directUri);
    if (token != null) return token;
  }

  final urlPattern = RegExp(r'https?://[^\s]+', caseSensitive: false);
  for (final match in urlPattern.allMatches(text)) {
    final uri = Uri.tryParse(match.group(0)!);
    if (uri == null) continue;
    final token = parseShareTokenFromUri(uri);
    if (token != null) return token;
  }
  return null;
}
