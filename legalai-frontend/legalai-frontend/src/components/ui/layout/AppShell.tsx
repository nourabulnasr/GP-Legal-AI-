import type { ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { logout } from "@/lib/auth";
import {
  FileSearch,
  History,
  LayoutDashboard,
  LogOut,
  MessageSquare,
  Plus,
  Scale,
  Search,
} from "lucide-react";

type User = { id: number; email: string; role: string };

type Props = {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  user: User;
  onLogout?: () => void;
  children: ReactNode;
};

const navItems = [
  { to: "/analyze", label: "Documents", icon: FileSearch },
  { to: "/history", label: "History", icon: History },
  { to: "/chat", label: "AI Chat", icon: MessageSquare },
  { to: "/admin", label: "Admin", icon: LayoutDashboard, adminOnly: true },
];

const breadcrumbLabels: Record<string, string> = {
  "/analyze": "Contract Analysis",
  "/history": "Analysis History",
  "/chat": "AI Chat Assistant",
  "/admin": "Admin Management",
};

function getBreadcrumb(pathname: string) {
  if (pathname === "/history") return "Pages / Analysis History";
  if (pathname === "/admin") return "Pages / Admin Management";
  if (pathname === "/chat") return "Projects / AI Chat Assistant";
  const label = breadcrumbLabels[pathname] ?? (pathname.slice(1) || "Documents");
  return `Documents / ${label}`;
}

export default function AppShell({
  title = "Legato",
  subtitle,
  right,
  user,
  onLogout,
  children,
}: Props) {
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    onLogout?.();
    navigate("/login", { replace: true });
  };

  const filteredNav = navItems.filter(
    (item) => !item.adminOnly || user.role === "admin"
  );

  const initials = user.email
    ? user.email.slice(0, 2).toUpperCase()
    : "U";

  return (
    <div className="min-h-screen bg-background flex">
      {/* Sidebar */}
      <aside className="w-60 border-r flex flex-col bg-card">
        <div className="p-4 border-b">
          <Link to="/analyze" className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-md bg-primary flex items-center justify-center">
              <Scale className="h-4 w-4 text-primary-foreground" />
            </div>
            <div>
              <span className="font-bold text-foreground block">Legato</span>
              <span className="text-xs text-muted-foreground">PRO SAAS</span>
            </div>
          </Link>
        </div>
        <nav className="flex-1 p-3 space-y-0.5">
          {filteredNav.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`
                }
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>
        {/* User profile at bottom */}
        <div className="p-3 border-t">
          <div className="flex items-center gap-3 px-2 py-2">
            <div className="h-9 w-9 rounded-full bg-primary/20 flex items-center justify-center text-sm font-semibold text-primary">
              {initials}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">
                {user.email}
              </p>
              <p className="text-xs text-muted-foreground capitalize">
                {user.role}
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-2 text-muted-foreground hover:text-destructive"
            onClick={handleLogout}
          >
            <LogOut className="h-4 w-4" />
            Logout
          </Button>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="border-b bg-background">
          <div className="px-6 py-3 flex items-center justify-between gap-4">
            <div>
              <p className="text-sm text-muted-foreground">
                {getBreadcrumb(location.pathname)}
              </p>
              <p className="text-sm font-medium text-foreground mt-0.5">{title}</p>
              {subtitle ? (
                <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
              ) : null}
            </div>
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  type="search"
                  placeholder="Search contracts..."
                  className="h-9 w-56 pl-9 pr-3 rounded-md border border-input bg-muted/50 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                />
              </div>
              <Link to="/analyze">
                <Button size="sm" className="gap-2">
                  <Plus className="h-4 w-4" />
                  New Analysis
                </Button>
              </Link>
              {right}
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-auto">
          <div className="mx-auto max-w-6xl px-6 py-6">{children}</div>
        </main>
      </div>
    </div>
  );
}
