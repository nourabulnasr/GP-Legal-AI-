import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

Future<void> downloadPdfFile(String filename, String content, {bool rtl = false}) async {
  if (kIsWeb) {
    throw UnsupportedError('Use pdf_download_web on web.');
  }
  final font = rtl
      ? await PdfGoogleFonts.notoSansArabicRegular()
      : await PdfGoogleFonts.notoSansRegular();
  final titleFont = rtl
      ? await PdfGoogleFonts.notoSansArabicBold()
      : await PdfGoogleFonts.notoSansBold();

  final doc = pw.Document();
  doc.addPage(
    pw.MultiPage(
      textDirection: rtl ? pw.TextDirection.rtl : pw.TextDirection.ltr,
      theme: pw.ThemeData.withFont(base: font, bold: titleFont),
      build: (context) => [
        pw.Text(
          rtl ? 'العقد المترجم' : 'Translated contract',
          style: pw.TextStyle(font: titleFont, fontSize: 18),
        ),
        pw.SizedBox(height: 12),
        pw.Text(content, style: const pw.TextStyle(fontSize: 11, lineSpacing: 1.4)),
      ],
    ),
  );

  final bytes = await doc.save();
  final name = filename.endsWith('.pdf') ? filename : '$filename.pdf';
  final path = '${Directory.systemTemp.path}${Platform.pathSeparator}$name';
  await File(path).writeAsBytes(bytes);
}
