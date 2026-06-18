import 'dart:typed_data';

void triggerFileDownload(Uint8List bytes, String filename, String mimeType) {
  // No-op on non-web platforms
}

void openFileInBrowser(Uint8List bytes, String mimeType) {
  // No-op on non-web platforms
}
