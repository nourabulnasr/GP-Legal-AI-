import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:legato_mobile/app_services.dart';

/// Dropdown that loads the user's analyses and lets them pick one by filename.
/// Replaces raw integer TextFields across all Phase 5 tool screens.
class AnalysisIdPicker extends StatefulWidget {
  const AnalysisIdPicker({
    super.key,
    required this.onChanged,
    this.label = 'Select analysis',
    this.enabled = true,
  });

  final ValueChanged<int?> onChanged;
  final String label;
  final bool enabled;

  @override
  State<AnalysisIdPicker> createState() => _AnalysisIdPickerState();
}

class _AnalysisIdPickerState extends State<AnalysisIdPicker> {
  bool _loading = true;
  List<_AnalysisEntry> _entries = const [];
  int? _selected;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final items = await context.read<AppServices>().legato.listAnalyses();
      if (!mounted) return;
      setState(() {
        _entries = items
            .whereType<Map>()
            .map((m) => _AnalysisEntry(
                  id: (m['id'] as num?)?.toInt() ?? 0,
                  filename: m['filename']?.toString() ?? 'Analysis ${m['id']}',
                  createdAt: m['created_at']?.toString() ?? '',
                ))
            .where((e) => e.id > 0)
            .toList();
        _loading = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const SizedBox(
        height: 56,
        child: Center(child: LinearProgressIndicator()),
      );
    }
    if (_entries.isEmpty) {
      return OutlinedButton.icon(
        icon: const Icon(Icons.refresh),
        label: const Text('No analyses — tap to retry'),
        onPressed: () {
          setState(() => _loading = true);
          _load();
        },
      );
    }
    return DropdownButtonFormField<int>(
      initialValue: _selected,
      decoration: InputDecoration(
        labelText: widget.label,
        border: const OutlineInputBorder(),
      ),
      isExpanded: true,
      items: _entries
          .map((e) => DropdownMenuItem<int>(
                value: e.id,
                child: Text(
                  e.filename,
                  overflow: TextOverflow.ellipsis,
                ),
              ))
          .toList(),
      onChanged: widget.enabled
          ? (v) {
              setState(() => _selected = v);
              widget.onChanged(v);
            }
          : null,
    );
  }
}

class _AnalysisEntry {
  const _AnalysisEntry({
    required this.id,
    required this.filename,
    required this.createdAt,
  });
  final int id;
  final String filename;
  final String createdAt;
}
