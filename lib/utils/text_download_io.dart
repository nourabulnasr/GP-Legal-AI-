import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

void downloadTextFile(String filename, String content) {
  if (kIsWeb) {
    throw UnsupportedError('Use text_download_web on web.');
  }
  final dir = Directory.systemTemp;
  final path = '${dir.path}${Platform.pathSeparator}$filename';
  File(path).writeAsStringSync(content, encoding: utf8);
  // Mobile/desktop: file is written to temp; callers may add share later.
}
