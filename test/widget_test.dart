import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:legato_mobile/main.dart';
import 'package:legato_mobile/providers/theme_notifier.dart';

void main() {
  testWidgets('Legato app builds', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({});
    final themeNotifier = await ThemeNotifier.init();
    await tester.pumpWidget(LegatoApp(themeNotifier: themeNotifier));
    await tester.pump();
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
