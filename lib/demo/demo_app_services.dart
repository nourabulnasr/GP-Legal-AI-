import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/demo/demo_api_client.dart';
import 'package:legato_mobile/demo/demo_auth_service.dart';
import 'package:legato_mobile/demo/demo_legato_api.dart';

/// [AppServices] variant that wires demo implementations instead of real ones.
class DemoAppServices extends AppServices {
  DemoAppServices() : super() {
    // AppServices() sets real implementations via late (non-final) fields.
    // Reassign them here with demo versions.
    api = DemoApiClient();
    auth = DemoAuthService();
    legato = DemoLegatoApi();
  }
}
