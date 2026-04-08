import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import AppShell from "@/components/ui/layout/AppShell";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { adminListAll, adminListUsers, adminUpdateUserRole, deleteAnalysis } from "@/lib/api";
import type { AnalysisDetail, AdminUser } from "@/lib/api";
import {
  Download,
  Plus,
  Filter,
  LayoutList,
  ChevronDown,
  MoreVertical,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  ShieldCheck,
  Gauge,
} from "lucide-react";

type Props = {
  user: { id: number; email: string; role: string };
  onLogout?: () => void;
};

const PAGE_SIZE = 10;

const USER_COLUMNS = ["User Details", "Role", "Tier", "Last Active", "Status"] as const;
const ANALYSIS_COLUMNS = ["Contract", "User", "Date", "Status"] as const;

function formatDate(s: string) {
  try {
    const d = new Date(s);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return s;
  }
}

function formatRelative(s: string) {
  try {
    const d = new Date(s);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    if (diffMins < 60) return `${diffMins} mins ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    if (diffDays < 7) return `${diffDays} days ago`;
    return formatDate(s);
  } catch {
    return s;
  }
}

type DerivedUser = AdminUser & {
  tier: string;
  lastActive: string;
  status: string;
};

function mergeUsersWithAnalyses(
  realUsers: AdminUser[],
  analyses: AnalysisDetail[],
  currentUser: { id: number; email: string; role: string }
): DerivedUser[] {
  const byUser = new Map<number, string>();
  for (const a of analyses) {
    const created = a.created_at;
    const prev = byUser.get(a.user_id) ?? "";
    if (!prev || created > prev) byUser.set(a.user_id, created);
  }
  return realUsers.map((u) => ({
    ...u,
    role: u.id === currentUser.id && currentUser.role === "admin" ? "Administrator" : u.role === "admin" ? "Administrator" : "User",
    tier: u.role === "admin" ? "ENTERPRISE" : "BASIC",
    lastActive: formatRelative(byUser.get(u.id) ?? ""),
    status: "Active",
  }));
}

export default function AdminPage({ user, onLogout }: Props) {
  const navigate = useNavigate();
  const [analyses, setAnalyses] = useState<AnalysisDetail[]>([]);
  const [realUsers, setRealUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [roleUpdating, setRoleUpdating] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"users" | "analyses" | "subscribers">("users");
  const [filterText, setFilterText] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [usersPage, setUsersPage] = useState(1);
  const [analysesPage, setAnalysesPage] = useState(1);
  const [roleFilter, setRoleFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [visibleUserCols, setVisibleUserCols] = useState<Set<string>>(new Set(USER_COLUMNS));
  const [visibleAnalysisCols, setVisibleAnalysisCols] = useState<Set<string>>(new Set(ANALYSIS_COLUMNS));

  const toggleUserCol = (col: string) => {
    setVisibleUserCols((prev) => {
      const n = new Set(prev);
      if (n.has(col)) n.delete(col);
      else n.add(col);
      return n;
    });
  };
  const toggleAnalysisCol = (col: string) => {
    setVisibleAnalysisCols((prev) => {
      const n = new Set(prev);
      if (n.has(col)) n.delete(col);
      else n.add(col);
      return n;
    });
  };

  const fetchAll = async () => {
    setLoading(true);
    setErr(null);
    try {
      const [analysesData, usersData] = await Promise.all([adminListAll(), adminListUsers()]);
      setAnalyses(analysesData);
      setRealUsers(usersData);
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "response" in e
          ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      setErr(typeof msg === "string" ? msg : "Failed to load analyses");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAll();
  }, []);

  const derivedUsers = useMemo(
    () => mergeUsersWithAnalyses(realUsers, analyses, user),
    [realUsers, analyses, user]
  );

  const filteredUsers = useMemo(() => {
    let out = derivedUsers;
    if (filterText.trim()) {
      const q = filterText.toLowerCase();
      out = out.filter(
        (u) =>
          u.email.toLowerCase().includes(q) ||
          u.role.toLowerCase().includes(q) ||
          String(u.id).includes(q)
      );
    }
    if (roleFilter !== "all") {
      const r = roleFilter.toLowerCase();
      out = out.filter((u) => {
        const ur = (u.role ?? "").toLowerCase();
        return r === "admin" ? (ur === "admin" || ur === "administrator") : ur === r;
      });
    }
    if (statusFilter !== "all") {
      const s = statusFilter.toLowerCase();
      out = out.filter((u) => u.status.toLowerCase() === s);
    }
    return out;
  }, [derivedUsers, filterText, roleFilter, statusFilter]);

  const filteredAnalyses = useMemo(() => {
    if (!filterText.trim()) return analyses;
    const q = filterText.toLowerCase();
    return analyses.filter(
      (a) =>
        (a.filename ?? "").toLowerCase().includes(q) ||
        String(a.user_id).includes(q) ||
        String(a.id).includes(q)
    );
  }, [analyses, filterText]);

  const paginatedUsers = useMemo(() => {
    const start = (usersPage - 1) * PAGE_SIZE;
    return filteredUsers.slice(start, start + PAGE_SIZE);
  }, [filteredUsers, usersPage]);

  const paginatedAnalyses = useMemo(() => {
    const start = (analysesPage - 1) * PAGE_SIZE;
    return filteredAnalyses.slice(start, start + PAGE_SIZE);
  }, [filteredAnalyses, analysesPage]);

  const usersTotalPages = Math.max(1, Math.ceil(filteredUsers.length / PAGE_SIZE));
  const analysesTotalPages = Math.max(1, Math.ceil(filteredAnalyses.length / PAGE_SIZE));

  const toggleSelectAll = (checked: boolean) => {
    if (checked) {
      const ids =
        activeTab === "users"
          ? paginatedUsers.map((u) => u.id)
          : paginatedAnalyses.map((a) => a.id);
      setSelectedIds(new Set(ids));
    } else {
      setSelectedIds(new Set());
    }
  };

  const toggleSelect = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const exportCsv = (onlySelected = false) => {
    const rows =
      activeTab === "users"
        ? (onlySelected && selectedIds.size > 0
            ? filteredUsers.filter((u) => selectedIds.has(u.id))
            : filteredUsers
          ).map((u) => [u.id, u.email, u.role, u.tier, u.lastActive, u.status])
        : (onlySelected && selectedIds.size > 0
            ? filteredAnalyses.filter((a) => selectedIds.has(a.id))
            : filteredAnalyses
          ).map((a) => [a.id, a.user_id, a.filename ?? "", formatDate(a.created_at)]);
    const headers =
      activeTab === "users"
        ? ["ID", "Email", "Role", "Tier", "Last Active", "Status"]
        : ["ID", "User ID", "Filename", "Date"];
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `admin-${activeTab}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getInitials = (email: string) => {
    const part = email.split("@")[0];
    return part
      .replace(/[^a-z0-9]/gi, "")
      .slice(0, 2)
      .toUpperCase() || "U";
  };

  const tierColors: Record<string, string> = {
    ENTERPRISE: "bg-primary/20 text-primary border-primary/30",
    PROFESSIONAL: "bg-purple-500/20 text-purple-700 border-purple-500/30",
    BASIC: "bg-muted text-muted-foreground border-muted-foreground/30",
  };

  return (
    <AppShell
      user={user}
      onLogout={onLogout}
      title="Admin Management"
      subtitle="Oversee platform users, subscriptions, and automated analysis logs."
    >
      <div className="space-y-6">
        {/* Header actions */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold">Admin Management</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Oversee platform users, subscriptions, and automated analysis logs.
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="gap-2" onClick={() => exportCsv()}>
              <Download className="h-4 w-4" />
              Export CSV
            </Button>
            <Button size="sm" className="gap-2" onClick={() => navigate("/register")}>
              <Plus className="h-4 w-4" />
              Add New User
            </Button>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b">
          <nav className="flex gap-6">
            {(["users", "analyses", "subscribers"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`pb-3 text-sm font-medium capitalize border-b-2 transition-colors ${
                  activeTab === tab
                    ? "border-primary text-primary"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>

        {/* Filter bar */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder={
                activeTab === "users"
                  ? "Filter by name or email..."
                  : "Filter by filename or user ID..."
              }
              value={filterText}
              onChange={(e) => {
                setFilterText(e.target.value);
                setUsersPage(1);
                setAnalysesPage(1);
              }}
              className="h-9 w-full pl-10 pr-4 rounded-md border border-input bg-background text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                {roleFilter === "all" ? "All Roles" : roleFilter}
                <ChevronDown className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" onCloseAutoFocus={(e) => e.preventDefault()}>
              <DropdownMenuItem
                onSelect={(e) => { e.preventDefault(); setRoleFilter("all"); }}
                onClick={() => setRoleFilter("all")}
              >
                All Roles
              </DropdownMenuItem>
              <DropdownMenuItem
                onSelect={(e) => { e.preventDefault(); setRoleFilter("user"); }}
                onClick={() => setRoleFilter("user")}
              >
                User
              </DropdownMenuItem>
              <DropdownMenuItem
                onSelect={(e) => { e.preventDefault(); setRoleFilter("admin"); }}
                onClick={() => setRoleFilter("admin")}
              >
                Admin
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                {statusFilter === "all" ? "All Status" : statusFilter}
                <ChevronDown className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuItem onSelect={() => setStatusFilter("all")} onClick={() => setStatusFilter("all")}>
                All Status
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => setStatusFilter("active")} onClick={() => setStatusFilter("active")}>
                Active
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => setStatusFilter("inactive")} onClick={() => setStatusFilter("inactive")}>
                Inactive
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                <LayoutList className="h-4 w-4" />
                Columns
                <ChevronDown className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuLabel>Toggle columns</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {USER_COLUMNS.map((col) => (
                <DropdownMenuCheckboxItem
                  key={col}
                  checked={visibleUserCols.has(col)}
                  onCheckedChange={() => toggleUserCol(col)}
                >
                  {col}
                </DropdownMenuCheckboxItem>
              ))}
              <DropdownMenuSeparator />
              {ANALYSIS_COLUMNS.map((col) => (
                <DropdownMenuCheckboxItem
                  key={col}
                  checked={visibleAnalysisCols.has(col)}
                  onCheckedChange={() => toggleAnalysisCol(col)}
                >
                  {col}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" className="gap-2" disabled={selectedIds.size === 0}>
                <ChevronDown className="h-4 w-4" />
                Bulk Actions ({selectedIds.size})
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuLabel>Actions for {selectedIds.size} selected</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => exportCsv(true)}>Export selected</DropdownMenuItem>
              <DropdownMenuItem onSelect={() => alert("Delete API coming soon.")}>Delete</DropdownMenuItem>
              <DropdownMenuItem onSelect={() => alert("Assign role API coming soon.")}>Assign role</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Content */}
        {loading ? (
          <div className="rounded-lg border p-12 flex flex-col items-center justify-center gap-3 text-muted-foreground">
            <div className="h-10 w-10 rounded-full border-4 border-primary border-t-transparent animate-spin" aria-hidden />
            <p className="text-sm">Loading admin data…</p>
          </div>
        ) : err ? (
          <div className="rounded-lg border border-destructive/50 bg-destructive/5 p-6 text-destructive">
            {err}
          </div>
        ) : activeTab === "users" ? (
          <>
            <div className="rounded-lg border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow className="bg-muted/50 hover:bg-muted/50">
                    <TableHead className="w-12">
                      <Checkbox
                        checked={
                          paginatedUsers.length > 0 &&
                          paginatedUsers.every((u) => selectedIds.has(u.id))
                        }
                        onCheckedChange={(v) => toggleSelectAll(!!v)}
                      />
                    </TableHead>
                    <TableHead className="font-semibold">User Details</TableHead>
                    <TableHead className="font-semibold">Role</TableHead>
                    <TableHead className="font-semibold">Tier</TableHead>
                    <TableHead className="font-semibold">Last Active</TableHead>
                    <TableHead className="font-semibold">Status</TableHead>
                    <TableHead className="font-semibold text-right w-12">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedUsers.map((u) => (
                    <TableRow key={u.id}>
                      <TableCell>
                        <Checkbox
                          checked={selectedIds.has(u.id)}
                          onCheckedChange={() => toggleSelect(u.id)}
                        />
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-3">
                          <div className="h-9 w-9 rounded-full bg-primary/20 flex items-center justify-center text-sm font-semibold text-primary">
                            {getInitials(u.email)}
                          </div>
                          <div>
                            <p className="font-medium">
                              {u.email ? u.email.split("@")[0] : `User #${u.id}`}
                            </p>
                            <p className="text-sm text-muted-foreground">{u.email || `user_${u.id}@platform.local`}</p>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>{u.role}</TableCell>
                      <TableCell>
                        <span
                          className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${
                            tierColors[u.tier] ?? tierColors.BASIC
                          }`}
                        >
                          {u.tier}
                        </span>
                      </TableCell>
                      <TableCell className="text-muted-foreground">{u.lastActive}</TableCell>
                      <TableCell>
                        <span className="inline-flex items-center gap-2">
                          <span
                            className="h-2 w-2 rounded-full bg-emerald-500"
                            aria-hidden
                          />
                          <span className="text-sm">{u.status}</span>
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Actions">
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem
                              onSelect={async () => {
                                try {
                                  await navigator.clipboard.writeText(u.email || "");
                                  alert("Email copied to clipboard.");
                                } catch {
                                  window.open(`mailto:${u.email}`);
                                }
                              }}
                            >
                              Contact user (copy email)
                            </DropdownMenuItem>
                            <DropdownMenuItem onSelect={() => navigate(`/history?user_id=${u.id}`)}>
                              View analyses
                            </DropdownMenuItem>
                            {u.id !== user.id && (
                              <>
                                <DropdownMenuItem
                                  onSelect={async () => {
                                    if (!confirm(`Set ${u.email} as Admin?`)) return;
                                    setRoleUpdating(u.id);
                                    try {
                                      await adminUpdateUserRole(u.id, "admin");
                                      await fetchAll();
                                    } catch (e: unknown) {
                                      const msg = e && typeof e === "object" && "response" in e
                                        ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
                                        : null;
                                      alert(typeof msg === "string" ? msg : "Failed to update role");
                                    } finally {
                                      setRoleUpdating(null);
                                    }
                                  }}
                                  disabled={roleUpdating === u.id}
                                >
                                  Set as Admin
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onSelect={async () => {
                                    if (!confirm(`Set ${u.email} as User?`)) return;
                                    setRoleUpdating(u.id);
                                    try {
                                      await adminUpdateUserRole(u.id, "user");
                                      await fetchAll();
                                    } catch (e: unknown) {
                                      const msg = e && typeof e === "object" && "response" in e
                                        ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
                                        : null;
                                      alert(typeof msg === "string" ? msg : "Failed to update role");
                                    } finally {
                                      setRoleUpdating(null);
                                    }
                                  }}
                                  disabled={roleUpdating === u.id}
                                >
                                  Set as User
                                </DropdownMenuItem>
                              </>
                            )}
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {filteredUsers.length > PAGE_SIZE && (
                <div className="flex items-center justify-between px-4 py-3 border-t">
                  <p className="text-sm text-muted-foreground">
                    Showing {(usersPage - 1) * PAGE_SIZE + 1} to{" "}
                    {Math.min(usersPage * PAGE_SIZE, filteredUsers.length)} of {filteredUsers.length}{" "}
                    users
                  </p>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => setUsersPage((p) => Math.max(1, p - 1))}
                      disabled={usersPage <= 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    {Array.from({ length: Math.min(5, usersTotalPages) }, (_, i) => i + 1).map(
                      (p) => (
                        <Button
                          key={p}
                          variant={usersPage === p ? "default" : "outline"}
                          size="sm"
                          className="h-8 w-8 p-0"
                          onClick={() => setUsersPage(p)}
                        >
                          {p}
                        </Button>
                      ))}
                    {usersTotalPages > 5 && (
                      <>
                        <span className="px-2 text-muted-foreground">…</span>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-8"
                          onClick={() => setUsersPage(usersTotalPages)}
                        >
                          {usersTotalPages}
                        </Button>
                      </>
                    )}
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => setUsersPage((p) => Math.min(usersTotalPages, p + 1))}
                      disabled={usersPage >= usersTotalPages}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : activeTab === "analyses" ? (
          <>
            <div className="rounded-lg border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow className="bg-muted/50 hover:bg-muted/50">
                    <TableHead className="w-12">
                      <Checkbox
                        checked={
                          paginatedAnalyses.length > 0 &&
                          paginatedAnalyses.every((a) => selectedIds.has(a.id))
                        }
                        onCheckedChange={(v) => toggleSelectAll(!!v)}
                      />
                    </TableHead>
                    <TableHead className="font-semibold">ID</TableHead>
                    <TableHead className="font-semibold">User ID</TableHead>
                    <TableHead className="font-semibold">Filename</TableHead>
                    <TableHead className="font-semibold">Date</TableHead>
                    <TableHead className="font-semibold text-right w-12">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedAnalyses.map((a) => (
                    <TableRow key={a.id}>
                      <TableCell>
                        <Checkbox
                          checked={selectedIds.has(a.id)}
                          onCheckedChange={() => toggleSelect(a.id)}
                        />
                      </TableCell>
                      <TableCell>{a.id}</TableCell>
                      <TableCell>{a.user_id}</TableCell>
                      <TableCell>{a.filename ?? "—"}</TableCell>
                      <TableCell className="text-muted-foreground">
                        {formatDate(a.created_at)}
                      </TableCell>
                      <TableCell className="text-right">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Actions">
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onSelect={() => navigate(`/analyze?analysis=${a.id}`)}>
                              View analysis
                            </DropdownMenuItem>
                            <DropdownMenuItem onSelect={() => navigate("/history")}>
                              Open in history
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onSelect={() => {
                                const det = analyses.find((x) => x.id === a.id);
                                if (det?.result_json) {
                                  try {
                                    const blob = new Blob([det.result_json], { type: "application/json" });
                                    const url = URL.createObjectURL(blob);
                                    const el = document.createElement("a");
                                    el.href = url;
                                    el.download = `analysis-${a.id}.json`;
                                    el.click();
                                    URL.revokeObjectURL(url);
                                  } catch {
                                    alert("Export failed.");
                                  }
                                } else {
                                  alert("Analysis #" + a.id + " - no data to export.");
                                }
                              }}
                            >
                              Export report
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onSelect={async () => {
                                if (!confirm(`Delete analysis #${a.id}?`)) return;
                                try {
                                  await deleteAnalysis(a.id);
                                  fetchAll();
                                } catch {
                                  alert("Failed to delete.");
                                }
                              }}
                              className="text-destructive focus:text-destructive"
                            >
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {filteredAnalyses.length > PAGE_SIZE && (
                <div className="flex items-center justify-between px-4 py-3 border-t">
                  <p className="text-sm text-muted-foreground">
                    Showing {(analysesPage - 1) * PAGE_SIZE + 1} to{" "}
                    {Math.min(analysesPage * PAGE_SIZE, filteredAnalyses.length)} of{" "}
                    {filteredAnalyses.length} analyses
                  </p>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => setAnalysesPage((p) => Math.max(1, p - 1))}
                      disabled={analysesPage <= 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    {Array.from({ length: Math.min(5, analysesTotalPages) }, (_, i) => i + 1).map(
                      (p) => (
                        <Button
                          key={p}
                          variant={analysesPage === p ? "default" : "outline"}
                          size="sm"
                          className="h-8 w-8 p-0"
                          onClick={() => setAnalysesPage(p)}
                        >
                          {p}
                        </Button>
                      ))}
                    {analysesTotalPages > 5 && (
                      <>
                        <span className="px-2 text-muted-foreground">…</span>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-8"
                          onClick={() => setAnalysesPage(analysesTotalPages)}
                        >
                          {analysesTotalPages}
                        </Button>
                      </>
                    )}
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() =>
                        setAnalysesPage((p) => Math.min(analysesTotalPages, p + 1))
                      }
                      disabled={analysesPage >= analysesTotalPages}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="rounded-lg border p-12 text-center">
            <p className="text-muted-foreground">Subscriber management coming soon.</p>
          </div>
        )}

        {/* Summary cards */}
        <div className="grid gap-4 sm:grid-cols-3">
          <Card className="relative overflow-hidden">
            <CardContent className="p-4">
              <BarChart3 className="absolute right-4 top-4 h-8 w-8 text-primary/30" />
              <p className="text-xs font-medium uppercase text-muted-foreground">
                Total Analyses
              </p>
              <p className="text-2xl font-bold mt-1">
                {analyses.length.toLocaleString()}
                <span className="text-sm font-normal text-emerald-600 ml-1">+0%</span>
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                Contracts processed this month
              </p>
            </CardContent>
          </Card>
          <Card className="relative overflow-hidden">
            <CardContent className="p-4">
              <ShieldCheck className="absolute right-4 top-4 h-8 w-8 text-purple-500/30" />
              <p className="text-xs font-medium uppercase text-muted-foreground">
                Active Licenses
              </p>
              <p className="text-2xl font-bold mt-1">
                {derivedUsers.length}
                <span className="text-sm font-normal text-emerald-600 ml-1">+0</span>
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                Enterprise accounts active
              </p>
            </CardContent>
          </Card>
          <Card className="relative overflow-hidden">
            <CardContent className="p-4">
              <Gauge className="absolute right-4 top-4 h-8 w-8 text-orange-500/30" />
              <p className="text-xs font-medium uppercase text-muted-foreground">
                System Load
              </p>
              <p className="text-2xl font-bold mt-1">
                32%
                <span className="text-sm font-normal text-emerald-600 ml-1">Stable</span>
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                Current AI processing utilization
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
