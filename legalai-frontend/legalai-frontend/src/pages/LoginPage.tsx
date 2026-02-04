import { useState, useEffect } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { login, me } from "@/lib/auth";
import { api } from "@/lib/api";
import AuthLayout from "@/components/ui/layout/AuthLayout";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

const VERIFY_EMAIL_MSG = "Please verify your email first. Check your inbox for the verification code.";

type User = { id: number; email: string; role: string };

type Props = { onLogin?: (user: User) => void };

export default function LoginPage({ onLogin }: Props) {
  const nav = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [verificationCode, setVerificationCode] = useState("");
  const [verifyLoading, setVerifyLoading] = useState(false);
  const [verifySuccess, setVerifySuccess] = useState<string | null>(null);
  const [resendMessage, setResendMessage] = useState<string | null>(null);
  const showVerifyBlock = error === VERIFY_EMAIL_MSG;

  useEffect(() => {
    const err = searchParams.get("error");
    const messages: Record<string, string> = {
      google_not_configured: "Google SSO is not yet configured. Please sign in with email.",
      google_denied: "Google sign-in was cancelled or denied.",
      google_token_failed: "Google sign-in failed. Please try again or use email.",
      google_user_failed: "Could not fetch your Google profile. Please try again.",
      google_no_email: "Google did not provide an email. Please use another sign-in method.",
    };
    if (err && messages[err]) {
      setError(messages[err]);
      setSearchParams({}, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await login(email, password);
      const user = await me();
      onLogin?.(user);
      if (user.role === "admin") {
        nav("/admin");
      } else {
        nav("/analyze");
      }
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      const errStr = typeof msg === "string" ? msg : "Login failed";
      setError(errStr);
      // When login fails with "verify your email first", auto-send the code so user gets it without clicking Resend
      if (errStr === VERIFY_EMAIL_MSG && email.trim()) {
        try {
          await api.post("/auth/resend-verification", { email: email.trim().toLowerCase() });
          setResendMessage("A verification code was sent to your email.");
        } catch {
          // Ignore resend errors; user can still click "Resend code"
        }
      }
    } finally {
      setLoading(false);
    }
  };

  const resendCode = async () => {
    if (!email.trim()) return;
    setResendMessage(null);
    setError(null);
    try {
      await api.post("/auth/resend-verification", { email: email.trim().toLowerCase() });
      setResendMessage("A new code was sent to your email.");
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      setError(typeof msg === "string" ? msg : "Could not resend code");
    }
  };

  const submitVerify = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!email.trim() || !verificationCode.trim()) return;
    setVerifyLoading(true);
    setError(null);
    setVerifySuccess(null);
    setResendMessage(null);
    try {
      await api.post("/auth/verify-email", { email: email.trim().toLowerCase(), code: verificationCode.trim() });
      setVerifySuccess("Email verified. Signing you in…");
      setVerificationCode("");
      await login(email, password);
      const user = await me();
      onLogin?.(user);
      if (user.role === "admin") {
        nav("/admin");
      } else {
        nav("/analyze");
      }
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      setError(typeof msg === "string" ? msg : "Verification failed");
    } finally {
      setVerifyLoading(false);
    }
  };

  return (
    <AuthLayout>
      <Card className="w-full max-w-[440px] shadow-lg rounded-xl border-0">
        <CardContent className="p-8">
          <h1 className="text-2xl font-bold text-foreground mb-1">
            Welcome back
          </h1>
          <p className="text-sm text-muted-foreground mb-6">
            Enter your credentials to access your legal workspace
          </p>

          <form onSubmit={submit} className="space-y-5">
            <div className="space-y-2">
              <label htmlFor="email" className="text-sm font-medium text-foreground">
                Email address
              </label>
              <Input
                id="email"
                type="email"
                placeholder="name@firm.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                required
                className="w-full rounded-md h-10"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label htmlFor="password" className="text-sm font-medium text-foreground">
                  Password
                </label>
                <Link
                  to="/forgot-password"
                  className="text-sm text-primary hover:underline"
                >
                  Forgot password?
                </Link>
              </div>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                required
                className="w-full rounded-md h-10"
              />
            </div>

            {error && (
              <div className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">
                {error}
              </div>
            )}

            {showVerifyBlock && (
              <div className="space-y-2 rounded-md border border-border bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  Enter the 6-character code from your email
                </p>
                <div className="flex gap-2">
                  <Input
                    type="text"
                    placeholder="e.g. a1b2c3"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value)}
                    onKeyDown={(e) => {
                      // Avoid submitting the outer login form when pressing Enter in the code field
                      if (e.key === "Enter") {
                        e.preventDefault();
                        submitVerify();
                      }
                    }}
                    maxLength={6}
                    autoComplete="one-time-code"
                    className="flex-1 rounded-md h-10 font-mono"
                  />
                  <Button
                    type="button"
                    onClick={() => submitVerify()}
                    disabled={verifyLoading || !verificationCode.trim()}
                    className="rounded-md h-10 shrink-0"
                  >
                    {verifyLoading ? "Verifying…" : "Verify email"}
                  </Button>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="text-sm text-muted-foreground"
                  onClick={resendCode}
                >
                  Resend code
                </Button>
                {resendMessage && (
                  <p className="text-sm text-green-600 dark:text-green-400">{resendMessage}</p>
                )}
              </div>
            )}

            {verifySuccess && (
              <div className="text-sm text-green-600 dark:text-green-400 bg-green-500/10 rounded-md px-3 py-2">
                {verifySuccess}
              </div>
            )}

            <Button
              type="submit"
              disabled={loading}
              className="w-full h-10 rounded-md"
            >
              {loading ? "Signing in..." : "Sign in to Legato"}
            </Button>
          </form>

          <div className="relative my-6">
            <Separator />
            <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-card px-2 text-xs text-muted-foreground uppercase tracking-wider">
              Or continue with
            </span>
          </div>

          <Button
            type="button"
            variant="outline"
            className="w-full h-10 rounded-md gap-2"
            onClick={() => window.location.assign(`${import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"}/auth/google`)}
          >
            <svg className="h-4 w-4" viewBox="0 0 24 24" aria-hidden>
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            Google SSO
          </Button>

          <p className="text-center text-sm text-muted-foreground mt-6">
            Don&apos;t have an account?{" "}
            <Link to="/register" className="font-medium text-primary hover:underline">
              Get started
            </Link>
          </p>
        </CardContent>
      </Card>
    </AuthLayout>
  );
}
