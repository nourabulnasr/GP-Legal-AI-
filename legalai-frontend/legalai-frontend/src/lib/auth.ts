import { api } from "@/lib/api";

export async function login(email: string, password: string) {
  const res = await api.post("/auth/login", {
    email,
    password,
  });

  const token = res.data?.access_token;
  if (!token) throw new Error("No access token returned");

  localStorage.setItem("access_token", token);
  return res.data;
}

export async function register(email: string, password: string) {
  const res = await api.post("/auth/register", {
    email,
    password,
  });
  return res.data;
}

export async function me() {
  const res = await api.get("/auth/me");
  return res.data;
}

export function logout() {
  localStorage.removeItem("access_token");
}


// File: src/lib/auth.ts — Quick summary 
// Purpose: Simple client-side auth helpers that call backend /auth/* endpoints via the shared api instance and manage an access token in localStorage.
// Exports: login, register, me, logout.
// How each function works — step‑by‑step
// import { api } from "@/lib/api";
// api is presumably an Axios (or similar) HTTP client configured with the backend base URL and (likely) interceptors that attach the Authorization header from localStorage token. The auth helpers rely on that behavior.
// async function login(email: string, password: string)
// Sends POST to /auth/login with { email, password }.
// Reads res.data?.access_token from the server response.
// If no token is present → throws new Error("No access token returned").
// If token exists → stores it in localStorage as "access_token" and returns res.data.
// Typical server response shape expected: { access_token: string, ...userData }.
// Key behavior: it both authenticates (server call) and persists the token locally.

// async function register(email: string, password: string)
// Sends POST to /auth/register with { email, password }.
// Returns whatever res.data the server returns.
// Note: This function does NOT save any token. If the server returns an access token on registration, the file does not store it — the caller must handle login or token storage.
// async function me()
// Sends GET to /auth/me.
// Returns res.data (usually the authenticated user object).
// Relies on api sending the current token (Authorization header) or the backend using cookies.
// function logout()
// Removes "access_token" from localStorage.
// Simple and synchronous.
// Important details, caveats & edge cases 
// No try/catch inside the helpers — errors from network or server will bubble up to the caller. Callers must catch errors.
// Token is stored in localStorage (accessible from JS) — vulnerable to XSS attacks. Not as secure as HttpOnly cookies.
// me() will fail if the Authorization header is not attached or token is expired; api must be configured to read token from localStorage.
// register() doesn't persist a token; UX should either log the user in after register or instruct caller to call login.
// No handling of token expiration, refresh tokens, or logout propagation across tabs.
// // Suggested improvements / hardening 
// Use constants for the key (e.g., TOKEN_KEY = "access_token").
// Add types to responses and return values for safer TypeScript usage.
// Add try/catch to return clearer errors or normalize errors to a common shape.
// Consider more secure storage — use HttpOnly cookies or a refresh-token flow to mitigate XSS risk.
// Provide helper getToken() and an isAuthenticated() utility.
// Add logic to handle refresh tokens or automatic logout on 401 responses (axios interceptor).
// // How to use these functions in the app 
// await login(email, pwd) → will store token and return server data. Then redirect or fetch me().
// await register(...) → check server response; if it contains a token, call login or store token explicitly.
// await me() → call on app bootstrap to load current user (but ensure api attaches token).
// logout() → call on sign-out to remove token and redirect to login.
// Summary: auth.ts is a small, straightforward wrapper around HTTP auth endpoints that persists an access token in localStorage. It works well for simple setups but should be extended for production security (token refresh, safer storage, typed responses, and robust error handling). ✅