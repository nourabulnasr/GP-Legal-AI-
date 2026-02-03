import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";

/**
 * Handles redirect from backend after Google OAuth.
 * Backend sends: /auth/google/callback?token=<jwt>
 * We store the token and reload to trigger App's me() check.
 */
export default function GoogleCallbackPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");
  const error = searchParams.get("error");

  useEffect(() => {
    if (error) {
      window.location.href = `/login?error=${encodeURIComponent(error)}`;
      return;
    }
    if (token) {
      localStorage.setItem("access_token", token);
      window.location.href = "/analyze";
      return;
    }
    window.location.href = "/login?error=google_no_token";
  }, [token, error]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <p className="text-muted-foreground">Signing you in...</p>
    </div>
  );
}
