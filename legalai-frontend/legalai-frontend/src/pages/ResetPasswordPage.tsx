import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import AuthLayout from "@/components/ui/layout/AuthLayout";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token") ?? "";
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    if (!token) {
      setError("Invalid reset link. Please request a new one.");
      return;
    }
    setLoading(true);
    try {
      await api.post("/auth/reset-password", {
        token,
        new_password: password,
      });
      setDone(true);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      setError(typeof msg === "string" ? msg : "Reset failed. The link may have expired.");
    } finally {
      setLoading(false);
    }
  };

  if (!token) {
    return (
      <AuthLayout>
        <Card className="w-full max-w-[440px] shadow-lg rounded-xl border-0">
          <CardContent className="p-8">
            <h1 className="text-2xl font-bold text-foreground mb-1">Invalid link</h1>
            <p className="text-sm text-muted-foreground mb-6">
              This password reset link is invalid or has expired. Please request a new one.
            </p>
            <Link to="/forgot-password">
              <Button className="w-full">Request new reset link</Button>
            </Link>
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

  if (done) {
    return (
      <AuthLayout>
        <Card className="w-full max-w-[440px] shadow-lg rounded-xl border-0">
          <CardContent className="p-8">
            <h1 className="text-2xl font-bold text-foreground mb-1">Password reset</h1>
            <p className="text-sm text-muted-foreground mb-6">
              Your password has been reset. You can now sign in with your new password.
            </p>
            <Link to="/login">
              <Button className="w-full">Sign in</Button>
            </Link>
          </CardContent>
        </Card>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout>
      <Card className="w-full max-w-[440px] shadow-lg rounded-xl border-0">
        <CardContent className="p-8">
          <h1 className="text-2xl font-bold text-foreground mb-1">Set new password</h1>
          <p className="text-sm text-muted-foreground mb-6">
            Enter your new password below.
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="password" className="text-sm font-medium text-foreground">
                New password
              </label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
                required
                minLength={6}
                className="w-full rounded-md h-10"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="confirm" className="text-sm font-medium text-foreground">
                Confirm password
              </label>
              <Input
                id="confirm"
                type="password"
                placeholder="••••••••"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                autoComplete="new-password"
                required
                minLength={6}
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
              {loading ? "Resetting..." : "Reset password"}
            </Button>
          </form>

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
