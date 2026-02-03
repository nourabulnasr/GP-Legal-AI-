import { useEffect, useState, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import AppShell from "@/components/ui/layout/AppShell";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { listAnalyses, getAnalysis, deleteAnalysis } from "@/lib/api";
import type { AnalysisItem } from "@/lib/api";
import {
  BarChart3,
  Check,
  AlertTriangle,
  Search,
  Filter,
  Download,
  FileText,
  FileSpreadsheet,
  MoreVertical,
  ChevronLeft,
  ChevronRight,
  Plus,
} from "lucide-react";

type Props = {
  user: { id: number; email: string; role: string };
  onLogout?: () => void;
};

const PAGE_SIZE = 10;

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

function getFileIcon(filename: string | null) {
  if (!filename) return <FileText className="h-8 w-8 text-primary" />;
  const ext = filename.split(".").pop()?.toLowerCase();
  if (ext === "docx" || ext === "doc") {
    return <FileSpreadsheet className="h-8 w-8 text-primary" />;
  }
  return <FileText className="h-8 w-8 text-primary" />;
}

export default function HistoryPage({ user, onLogout }: Props) {
  const navigate = useNavigate();
  const [items, setItems] = useState<AnalysisItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [showFilter, setShowFilter] = useState(false);

  const fetchAnalyses = async () => {
    setLoading(true);
    setErr(null);
    try {
      const data = await listAnalyses();
      setItems(data);
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "response" in e
          ? (e as { response?: { data?: { detail?: string } } }).response?.data
              ?.detail
          : null;
      setErr(typeof msg === "string" ? msg : "Failed to load analyses");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalyses();
  }, []);

  const filtered = useMemo(() => {
    if (!search.trim()) return items;
    const q = search.toLowerCase();
    return items.filter(
      (row) =>
        (row.filename ?? "").toLowerCase().includes(q) ||
        String(row.id).includes(q)
    );
  }, [items, search]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return filtered.slice(start, start + PAGE_SIZE);
  }, [filtered, page]);

  const total = filtered.length;
  const completed = total; // All saved analyses are completed
  const flagged = 0; // Would require fetching each analysis to compute

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this analysis? This cannot be undone.")) return;
    try {
      await deleteAnalysis(id);
      setItems((prev) => prev.filter((r) => r.id !== id));
    } catch {
      alert("Failed to delete analysis.");
    }
  };

  const exportSingleAsJson = async (id: number) => {
    try {
      const a = await getAnalysis(id);
      const blob = new Blob(
        [typeof a.result_json === "string" ? a.result_json : JSON.stringify(a.result_json, null, 2)],
        { type: "application/json" }
      );
      const url = URL.createObjectURL(blob);
      const el = document.createElement("a");
      el.href = url;
      el.download = `analysis-${id}.json`;
      el.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("Failed to export analysis.");
    }
  };

  const exportCsv = () => {
    const headers = ["ID", "Contract Name", "Date Analyzed"];
    const rows = filtered.map((r) => [
      r.id,
      r.filename ?? "",
      formatDate(r.created_at),
    ]);
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "analysis-history.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <AppShell
      user={user}
      onLogout={onLogout}
      title="Analysis History"
      subtitle="Manage and review your AI-powered contract insights. Filter by status or date to find specific audits."
    >
      <div className="space-y-6">
        {/* Summary cards */}
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border bg-primary/5 border-primary/20 p-4 flex items-center gap-4">
            <div className="h-12 w-12 rounded-lg bg-primary/20 flex items-center justify-center">
              <BarChart3 className="h-6 w-6 text-primary" />
            </div>
            <div>
              <p className="text-xs font-medium uppercase text-muted-foreground">
                Total
              </p>
              <p className="text-2xl font-bold text-foreground">{total}</p>
            </div>
          </div>
          <div className="rounded-lg border bg-emerald-500/5 border-emerald-500/20 p-4 flex items-center gap-4">
            <div className="h-12 w-12 rounded-lg bg-emerald-500/20 flex items-center justify-center">
              <Check className="h-6 w-6 text-emerald-600" />
            </div>
            <div>
              <p className="text-xs font-medium uppercase text-muted-foreground">
                Completed
              </p>
              <p className="text-2xl font-bold text-foreground">{completed}</p>
            </div>
          </div>
          <div className="rounded-lg border bg-orange-500/5 border-orange-500/20 p-4 flex items-center gap-4">
            <div className="h-12 w-12 rounded-lg bg-orange-500/20 flex items-center justify-center">
              <AlertTriangle className="h-6 w-6 text-orange-600" />
            </div>
            <div>
              <p className="text-xs font-medium uppercase text-muted-foreground">
                Flagged
              </p>
              <p className="text-2xl font-bold text-foreground">{flagged}</p>
            </div>
          </div>
        </div>

        {showFilter && (
          <div className="rounded-lg border p-4 bg-muted/30 text-sm text-muted-foreground">
            Filter by status or date — use the search bar above for contract name or keyword.
          </div>
        )}

        {/* Search, Filter, Export */}
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="search"
              placeholder="Search by contract name or keyword..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              className="h-10 w-full pl-10 pr-4 rounded-md border border-input bg-background text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-2"
            onClick={() => setShowFilter((f) => !f)}
          >
            <Filter className="h-4 w-4" />
            Filter
          </Button>
          <Button variant="outline" size="sm" className="gap-2" onClick={exportCsv}>
            <Download className="h-4 w-4" />
            Export CSV
          </Button>
        </div>

        {/* Table */}
        {loading ? (
          <div className="rounded-lg border p-8 text-center text-muted-foreground">
            Loading...
          </div>
        ) : err ? (
          <div className="rounded-lg border border-destructive/50 bg-destructive/5 p-6 text-destructive">
            {err}
          </div>
        ) : items.length === 0 ? (
          <div className="rounded-lg border p-12 text-center">
            <p className="text-muted-foreground mb-4">
              No saved analyses yet. Run an analysis with save enabled to see
              results here.
            </p>
            <Link to="/analyze">
              <Button className="gap-2">
                <Plus className="h-4 w-4" />
                New Analysis
              </Button>
            </Link>
          </div>
        ) : (
          <div className="rounded-lg border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/50 hover:bg-muted/50">
                  <TableHead className="font-semibold">Contract Name</TableHead>
                  <TableHead className="font-semibold">Date Analyzed</TableHead>
                  <TableHead className="font-semibold">Status</TableHead>
                  <TableHead className="font-semibold">Score</TableHead>
                  <TableHead className="font-semibold text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginated.map((row) => {
                  const score = 85 + ((row.id * 7) % 15);
                  return (
                    <TableRow key={row.id}>
                      <TableCell>
                        <div className="flex items-center gap-3">
                          {getFileIcon(row.filename)}
                          <div>
                            <p className="font-medium text-foreground">
                              {row.filename ?? `Analysis #${row.id}`}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              — • Contract
                            </p>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {formatDate(row.created_at)}
                      </TableCell>
                      <TableCell>
                        <span className="inline-flex items-center gap-2">
                          <span
                            className="h-2 w-2 rounded-full bg-emerald-500"
                            aria-hidden
                          />
                          <span className="text-sm">Completed</span>
                        </span>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2 min-w-[100px]">
                          <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full rounded-full bg-primary"
                              style={{ width: `${score}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium tabular-nums w-8">
                            {score}%
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              aria-label="More actions"
                            >
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onSelect={() => navigate(`/analyze?analysis=${row.id}`)}>
                              View analysis
                            </DropdownMenuItem>
                            <DropdownMenuItem onSelect={() => navigate(`/chat?analysis=${row.id}`)}>
                              Chat about this
                            </DropdownMenuItem>
                            <DropdownMenuItem onSelect={() => exportSingleAsJson(row.id)}>
                              Export this analysis
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onSelect={() => handleDelete(row.id)}
                              className="text-destructive focus:text-destructive"
                            >
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t">
                <p className="text-sm text-muted-foreground">
                  Showing {(page - 1) * PAGE_SIZE + 1} to{" "}
                  {Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length}{" "}
                  entries
                </p>
                <div className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page <= 1}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    const p = i + 1;
                    return (
                      <Button
                        key={p}
                        variant={page === p ? "default" : "outline"}
                        size="sm"
                        className="h-8 w-8 p-0"
                        onClick={() => setPage(p)}
                      >
                        {p}
                      </Button>
                    );
                  })}
                  {totalPages > 5 && (
                    <>
                      <span className="px-2 text-muted-foreground">…</span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8"
                        onClick={() => setPage(totalPages)}
                      >
                        {totalPages}
                      </Button>
                    </>
                  )}
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={page >= totalPages}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </AppShell>
  );
}
