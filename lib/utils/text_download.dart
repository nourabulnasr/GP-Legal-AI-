export 'text_download_stub.dart'
    if (dart.library.html) 'text_download_web.dart'
    if (dart.library.io) 'text_download_io.dart';
