import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:legato_mobile/main.dart';

void main() {
  testWidgets('Legato app builds', (WidgetTester tester) async {
    await tester.pumpWidget(const LegatoApp());
    await tester.pump();
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
