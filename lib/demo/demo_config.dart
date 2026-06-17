/// Set to true to run entirely without a backend.
/// When true, [DemoAppServices] is used instead of [AppServices],
/// all API calls return local mock data, and the app auto-logs in.
///
/// To switch persona, log out then log in with one of:
///   admin@demo.com   → admin dashboard
///   lawyer@demo.com  → pending-verification lawyer
///   verified@demo.com→ approved lawyer (checkmark shows)
///   rejected@demo.com→ rejected lawyer (rejection banner)
///   user@demo.com    → regular user (any other email also works)
///
/// Password is ignored in demo mode.
const bool kDemoMode = false;
