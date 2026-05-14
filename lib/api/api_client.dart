import 'dart:convert';
import 'dart:io' show HttpClient;
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import 'package:http/io_client.dart';
import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/config/app_config.dart';
import 'package:legato_mobile/config/runtime_config.dart';
import 'package:legato_mobile/storage/token_storage.dart';

/// Builds an [http.Client] with explicit TCP timeouts (native only; web uses the browser's fetch client).
http.Client _defaultHttpClient() {
  if (kIsWeb) return http.Client();
  final hc = HttpClient()
    ..connectionTimeout = const Duration(seconds: 30)
    ..idleTimeout = const Duration(seconds: 120);
  return IOClient(hc);
}

/// HTTP client mirroring `legalai-frontend/src/lib/api.ts` (axios instance + interceptors).
class ApiClient {
  ApiClient({
    http.Client? httpClient,
    TokenStorage? storage,
    void Function()? onUnauthorized,
  })  : _http = httpClient ?? _defaultHttpClient(),
        _storage = storage ?? TokenStorage(),
        _onUnauthorized = onUnauthorized;

  final http.Client _http;
  final TokenStorage _storage;
  final void Function()? _onUnauthorized;

  Uri uri(String path) => Uri.parse('${RuntimeConfig.apiBaseUrl}$path');

  Future<Map<String, String>> _headers({bool jsonBody = false}) async {
    final h = <String, String>{};
    if (jsonBody) {
      h['Content-Type'] = 'application/json; charset=utf-8';
    }
    final t = await _storage.readToken();
    if (t != null && t.isNotEmpty) {
      h['Authorization'] = 'Bearer $t';
    }
    return h;
  }

  Future<void> _on401(http.Response r) async {
    if (r.statusCode == 401) {
      await _storage.clearToken();
      _onUnauthorized?.call();
    }
  }

  String _extractDetail(String body) {
    try {
      final m = jsonDecode(body);
      if (m is Map && m['detail'] != null) {
        final d = m['detail'];
        if (d is String) return d;
        return jsonEncode(d);
      }
    } catch (_) {}
    return body.isEmpty ? 'Request failed' : body;
  }

  Future<Map<String, dynamic>> getJson(String path) async {
    final r = await _http
        .get(uri(path), headers: await _headers())
        .timeout(AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      if (r.body.isEmpty) return {};
      return Map<String, dynamic>.from(jsonDecode(r.body) as Map);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  /// GET with query string (e.g. `/legato/deal-threads?analysis_id=1`).
  Future<Map<String, dynamic>> getJsonQuery(String path, Map<String, String> query) async {
    final u = uri(path).replace(queryParameters: query);
    final r = await _http.get(u, headers: await _headers()).timeout(AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      if (r.body.isEmpty) return {};
      return Map<String, dynamic>.from(jsonDecode(r.body) as Map);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  Future<List<dynamic>> getJsonListQuery(String path, Map<String, String> query) async {
    final u = uri(path).replace(queryParameters: query);
    final r = await _http.get(u, headers: await _headers()).timeout(AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      return List<dynamic>.from(jsonDecode(r.body) as List);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  Future<List<dynamic>> getJsonList(String path) async {
    final r = await _http.get(uri(path), headers: await _headers()).timeout(AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      return List<dynamic>.from(jsonDecode(r.body) as List);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  Future<Map<String, dynamic>> postJson(
    String path,
    Map<String, dynamic> body, {
    Duration? timeout,
  }) async {
    final r = await _http
        .post(
          uri(path),
          headers: await _headers(jsonBody: true),
          body: jsonEncode(body),
        )
        .timeout(timeout ?? AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      if (r.body.isEmpty) return {};
      return Map<String, dynamic>.from(jsonDecode(r.body) as Map);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  Future<Map<String, dynamic>> patchJson(String path, Map<String, dynamic> body) async {
    final r = await _http
        .patch(
          uri(path),
          headers: await _headers(jsonBody: true),
          body: jsonEncode(body),
        )
        .timeout(AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      return Map<String, dynamic>.from(jsonDecode(r.body) as Map);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  Future<Map<String, dynamic>> putJson(String path, Map<String, dynamic> body) async {
    final r = await _http
        .put(
          uri(path),
          headers: await _headers(jsonBody: true),
          body: jsonEncode(body),
        )
        .timeout(AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      if (r.body.isEmpty) return {};
      return Map<String, dynamic>.from(jsonDecode(r.body) as Map);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  Future<Map<String, dynamic>> deleteJson(String path) async {
    final r = await _http.delete(uri(path), headers: await _headers()).timeout(AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      if (r.body.isEmpty) return {};
      return Map<String, dynamic>.from(jsonDecode(r.body) as Map);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  /// Long-running multipart — mirrors `analyzeContract` in `api.ts`.
  Future<Map<String, dynamic>> postMultipartOcrCheck({
    required Uint8List fileBytes,
    required String filename,
    bool useRag = true,
    bool useMl = true,
    bool useLlm = true,
    int? llmTopK,
    int? llmMaxNewTokens,
    bool save = true,
    String? query,
  }) async {
    final request = http.MultipartRequest('POST', uri('/ocr_check_and_search'));
    final t = await _storage.readToken();
    if (t != null && t.isNotEmpty) {
      request.headers['Authorization'] = 'Bearer $t';
    }
    request.files.add(
      http.MultipartFile.fromBytes('file', fileBytes, filename: filename),
    );
    request.fields['use_rag'] = useRag.toString();
    request.fields['use_ml'] = useMl.toString();
    request.fields['use_llm'] = useLlm.toString();
    if (llmTopK != null) request.fields['llm_top_k'] = '$llmTopK';
    if (llmMaxNewTokens != null) {
      request.fields['llm_max_new_tokens'] = '$llmMaxNewTokens';
    }
    request.fields['save'] = save.toString();
    if (query != null && query.isNotEmpty) {
      request.fields['query'] = query;
    }

    final streamed = await request.send().timeout(AppConfig.longTimeout);
    final response = await http.Response.fromStream(streamed);
    await _on401(response);
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return Map<String, dynamic>.from(jsonDecode(response.body) as Map);
    }
    throw ApiException(_extractDetail(response.body), statusCode: response.statusCode);
  }

  Future<Map<String, dynamic>> postMultipart(
    String path, {
    required String filePath,
    required String fieldName,
    required String filename,
    Map<String, String>? fields,
    Duration? timeout,
  }) async {
    final request = http.MultipartRequest('POST', uri(path));
    final t = await _storage.readToken();
    if (t != null && t.isNotEmpty) {
      request.headers['Authorization'] = 'Bearer $t';
    }
    (fields ?? const {}).forEach((k, v) => request.fields[k] = v);
    request.files.add(await http.MultipartFile.fromPath(fieldName, filePath, filename: filename));
    final streamed = await request.send().timeout(timeout ?? AppConfig.longTimeout);
    final response = await http.Response.fromStream(streamed);
    await _on401(response);
    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (response.body.isEmpty) return {};
      return Map<String, dynamic>.from(jsonDecode(response.body) as Map);
    }
    throw ApiException(_extractDetail(response.body), statusCode: response.statusCode);
  }

  Future<Map<String, dynamic>> postJsonLong(String path, Map<String, dynamic> body) async {
    final r = await _http
        .post(
          uri(path),
          headers: await _headers(jsonBody: true),
          body: jsonEncode(body),
        )
        .timeout(AppConfig.chatTimeout);
    await _on401(r);
    if (r.statusCode >= 200 && r.statusCode < 300) {
      return Map<String, dynamic>.from(jsonDecode(r.body) as Map);
    }
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }

  Future<String> getHealthRaw() async {
    final r = await _http.get(uri('/health')).timeout(AppConfig.defaultTimeout);
    await _on401(r);
    if (r.statusCode == 200) return r.body;
    throw ApiException(_extractDetail(r.body), statusCode: r.statusCode);
  }
}
