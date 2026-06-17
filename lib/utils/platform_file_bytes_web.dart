import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';

Future<Uint8List?> readPlatformFileBytes(PlatformFile file) async {
  if (file.bytes != null && file.bytes!.isNotEmpty) {
    return file.bytes;
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
