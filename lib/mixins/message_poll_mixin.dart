import 'dart:async';

import 'package:flutter/material.dart';

/// Polls a callback every [interval] while this [State] is mounted.
mixin MessagePollMixin<T extends StatefulWidget> on State<T> {
  Timer? _pollTimer;

  Duration get pollInterval => const Duration(seconds: 1);

  void startMessagePolling(Future<void> Function() onPoll) {
    stopMessagePolling();
    _pollTimer = Timer.periodic(pollInterval, (_) {
      if (!mounted) return;
      onPoll();
    });
  }

  void stopMessagePolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  @override
  void dispose() {
    stopMessagePolling();
    super.dispose();
  }
}
