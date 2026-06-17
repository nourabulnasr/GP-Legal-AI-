import 'dart:typed_data';

class PendingLawyerDocs {
  static Uint8List? cvBytes;
  static String? cvFilename;
  static Uint8List? idCardBytes;
  static String? idCardFilename;

  static bool get hasPending => cvBytes != null || idCardBytes != null;

  static void clear() {
    cvBytes = null;
    cvFilename = null;
    idCardBytes = null;
    idCardFilename = null;
  }
}
