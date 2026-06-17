import 'package:legato_mobile/api/api_client.dart';

/// Overrides only [getHealthRaw] so the dashboard shows "Connected" in demo mode.
class DemoApiClient extends ApiClient {
  DemoApiClient() : super();

  @override
  Future<String> getHealthRaw() async => 'ok';
}
