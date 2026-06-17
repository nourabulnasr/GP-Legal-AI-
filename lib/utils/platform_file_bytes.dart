import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';

import 'platform_file_bytes_io.dart'
    if (dart.library.html) 'platform_file_bytes_web.dart' as impl;

Future<Uint8List?> readPlatformFileBytes(PlatformFile file) =>
    impl.readPlatformFileBytes(file);

String pickedImageFilename(PlatformFile file) => impl.pickedImageFilename(file);
