/// Egypt (Africa/Cairo) time helpers — UTC+2 year-round.
library;

const egyptUtcOffset = Duration(hours: 2);
const _months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/// Current clock in Egypt, derived from UTC (Africa/Cairo has no DST).
DateTime nowInEgypt() => DateTime.now().toUtc().add(egyptUtcOffset);

DateTime? _parseUtc(String? iso) {
  if (iso == null || iso.isEmpty) return null;
  try {
    var raw = iso.trim();
    if (!raw.endsWith('Z') && !raw.contains('+') && !raw.contains('-', raw.indexOf('T') + 1) && raw.contains('T')) {
      raw = '${raw}Z';
    }
    return DateTime.parse(raw).toUtc();
  } catch (_) {
    return null;
  }
}

DateTime _toEgypt(DateTime utc) => utc.add(egyptUtcOffset);

String _formatEgyptDateTime(DateTime local, {bool timeOnly = false}) {
  final hour12 = local.hour % 12 == 0 ? 12 : local.hour % 12;
  final amPm = local.hour < 12 ? 'AM' : 'PM';
  final timeStr = '$hour12:${local.minute.toString().padLeft(2, '0')} $amPm';
  if (timeOnly) return timeStr;
  return '${local.day} ${_months[local.month - 1]} · $timeStr';
}

/// Convert Egypt-local calendar values (from date/time pickers) to UTC ISO for the API.
String egyptLocalDateTimeToUtcIso({
  required int year,
  required int month,
  required int day,
  required int hour,
  required int minute,
}) {
  final utc = DateTime.utc(year, month, day, hour, minute).subtract(egyptUtcOffset);
  return utc.toIso8601String();
}

bool isEgyptLocalDateTimeInFuture({
  required int year,
  required int month,
  required int day,
  required int hour,
  required int minute,
}) {
  final utc = DateTime.utc(year, month, day, hour, minute).subtract(egyptUtcOffset);
  return utc.isAfter(DateTime.now().toUtc());
}

/// Format a UTC ISO-8601 timestamp for display in Egyptian local time.
String formatConsultationTimeEgypt(String? iso, {bool timeOnly = false}) {
  final utc = _parseUtc(iso);
  if (utc == null) return iso ?? '—';
  return _formatEgyptDateTime(_toEgypt(utc), timeOnly: timeOnly);
}

/// Rewrite legacy notification text that embeds `YYYY-MM-DD HH:MM UTC`.
String rewriteUtcInNotificationMessage(String message) {
  final pattern = RegExp(r'on (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) UTC');
  return message.replaceAllMapped(pattern, (match) {
    final iso = '${match.group(1)}T${match.group(2)}:00Z';
    return 'at ${formatConsultationTimeEgypt(iso)}';
  });
}

/// Rewrite older `DD/MM/YYYY · HH:MM` labels (already Egypt local) to `18 Jun · 7:00 AM`.
String rewriteLegacyEgyptNotificationMessage(String message) {
  final pattern = RegExp(r'(?:on|at) (\d{1,2})/(\d{1,2})/(\d{4}) · (\d{2}):(\d{2})');
  return message.replaceAllMapped(pattern, (match) {
    final local = DateTime(
      int.parse(match.group(3)!),
      int.parse(match.group(2)!),
      int.parse(match.group(1)!),
      int.parse(match.group(4)!),
      int.parse(match.group(5)!),
    );
    return 'at ${_formatEgyptDateTime(local)}';
  });
}

String formatActivityMessage(String? message) {
  var text = message ?? 'Activity';
  text = rewriteUtcInNotificationMessage(text);
  text = rewriteLegacyEgyptNotificationMessage(text);
  return text;
}

String formatEgyptTimeOfDay(int hour, int minute) =>
    _formatEgyptDateTime(DateTime(2000, 1, 1, hour, minute), timeOnly: true);

String formatEgyptDate(DateTime d) => '${d.day}/${d.month}/${d.year}';
