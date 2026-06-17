/// Fallback when neither dart:html nor dart:io is available.
void downloadTextFile(String filename, String content) {
  throw UnsupportedError('Download is not supported on this platform.');
}
