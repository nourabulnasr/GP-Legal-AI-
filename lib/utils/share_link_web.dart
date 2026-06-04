import 'dart:html' as html;

void clearShareQueryFromUrl() {
  html.window.history.replaceState(null, '', '/');
}
