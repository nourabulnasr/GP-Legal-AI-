import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Scale } from "lucide-react";

type Props = { children: ReactNode };

export default function AuthLayout({ children }: Props) {
  return (
    <div className="min-h-screen flex flex-col bg-[#f5f5f5]">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-border/50 bg-background">
        <Link to="/" className="flex items-center gap-2 font-semibold text-foreground">
          <Scale className="h-6 w-6 text-primary" />
          Legato
        </Link>
        <div className="flex items-center gap-4">
          <a
            href="mailto:support@legato.local?subject=Help%20Center"
            className="text-sm font-medium text-muted-foreground hover:text-foreground"
          >
            Help Center
          </a>
          <a href="mailto:sales@legato.local?subject=Contact%20Sales">
            <Button size="sm" className="rounded-md">
              Contact Sales
            </Button>
          </a>
        </div>
      </header>

      {/* Main: centered card */}
      <main className="flex-1 flex items-center justify-center p-6">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 bg-background py-6 px-6">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-6 text-sm text-muted-foreground">
            <Link to="#" className="hover:text-foreground">Privacy Policy</Link>
            <Link to="#" className="hover:text-foreground">Terms of Service</Link>
            <Link to="#" className="hover:text-foreground">Security</Link>
          </div>
          <p className="text-sm text-muted-foreground">
            © 2025 Legato. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
