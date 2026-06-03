import 'dart:html' as html;
import 'dart:typed_data';

import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

Future<void> downloadPdfFile(String filename, String content, {bool rtl = false}) async {
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
  final blob = html.Blob([Uint8List.fromList(bytes)], 'application/pdf');
  final url = html.Url.createObjectUrlFromBlob(blob);
  final anchor = html.AnchorElement(href: url)
    ..setAttribute('download', filename.endsWith('.pdf') ? filename : '$filename.pdf')
    ..style.display = 'none';
  html.document.body?.children.add(anchor);
  anchor.click();
  anchor.remove();
  html.Url.revokeObjectUrl(url);
}
