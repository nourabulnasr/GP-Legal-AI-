import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthLayout from "@/components/ui/layout/AuthLayout";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

export default function ForgotPasswordPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRequestCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await api.post("/auth/forgot-password", { email: email.trim().toLowerCase() });
      setSent(true);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      setError(typeof msg === "string" ? msg : "Request failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!code.trim()) {
      setError("Enter the code from your email.");
      return;
    }
    setLoading(true);
    try {
      const res = await api.post<{ reset_token: string }>("/auth/verify-reset-code", {
        email: email.trim().toLowerCase(),
        code: code.trim(),
      });
      const resetToken = res.data?.reset_token;
      if (resetToken) {
        navigate(`/reset-password?token=${encodeURIComponent(resetToken)}`);
      } else {
        setError("Invalid response. Please try again.");
      }
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      setError(typeof msg === "string" ? msg : "Invalid or expired code. Request a new one.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout>
      <Card className="w-full max-w-[440px] shadow-lg rounded-xl border-0">
        <CardContent className="p-8">
          <h1 className="text-2xl font-bold text-foreground mb-1">Forgot password?</h1>
          <p className="text-sm text-muted-foreground mb-6">
            {sent
              ? "Check your email for a 6-character code, then enter it below."
              : "Enter your email and we'll send you a code to reset your password."}
          </p>

          {sent ? (
            <form onSubmit={handleVerifyCode} className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="code" className="text-sm font-medium text-foreground">
                  Verification code
                </label>
                <Input
                  id="code"
                  type="text"
                  placeholder="e.g. a1b2c3"
                  value={code}
                  onChange={(e) => setCode(e.target.value.replace(/\s/g, "").slice(0, 6))}
                  autoComplete="one-time-code"
                  className="w-full rounded-md h-10 font-mono tracking-widest"
                />
              </div>
              {error && (
                <div className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">
                  {error}
                </div>
              )}
              <Button type="submit" disabled={loading} className="w-full h-10">
                {loading ? "Verifying..." : "Continue to reset password"}
              </Button>
              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => setSent(false)}
                disabled={loading}
              >
                Use a different email
              </Button>
            </form>
          ) : (
            <form onSubmit={handleRequestCode} className="space-y-4">
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
              {error && (
                <div className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">
                  {error}
                </div>
              )}
              <Button
                type="submit"
                disabled={loading}
                className="w-full h-10 rounded-md"
              >
                {loading ? "Sending..." : "Send code"}
              </Button>
            </form>
          )}

          <p className="text-center text-sm text-muted-foreground mt-6">
            <Link to="/login" className="text-primary hover:underline">
              Back to sign in
            </Link>
          </p>
        </CardContent>
      </Card>
    </AuthLayout>
  );
}
