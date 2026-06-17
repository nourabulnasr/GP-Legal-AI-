import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/lib/api";

/**
 * Handles redirect from backend after Google OAuth.
 * Backend sends: /auth/google/callback?token=<jwt>
 * We store the token, fetch user to determine role, then redirect.
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
      api
        .get<{ id: number; email: string; role: string }>("/auth/me")
        .then((res) => {
          const user = res.data;
          window.location.href = user?.role === "admin" ? "/admin" : "/analyze";
        })
        .catch(() => {
          window.location.href = "/analyze";
        });
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
