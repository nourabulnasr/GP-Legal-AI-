import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';

Future<Uint8List?> readPlatformFileBytes(PlatformFile file) async {
  if (file.bytes != null && file.bytes!.isNotEmpty) {
    return file.bytes;
  }
  final path = file.path;
  if (path != null && path.isNotEmpty) {
    final data = await File(path).readAsBytes();
    if (data.isNotEmpty) return data;
  }
  return null;
}

String pickedImageFilename(PlatformFile file) {
  final name = file.name.trim();
  if (name.isNotEmpty && name.contains('.')) return name;
  final ext = file.extension?.trim();
  if (ext != null && ext.isNotEmpty) return 'image.$ext';
  return 'image.jpg';
}
