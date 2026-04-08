import { useMemo, useState, useRef, useCallback, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import AppShell from "@/components/ui/layout/AppShell";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { analyzeContract, saveAnalysisToDb, getAnalysis, chatWithDocument, type AnalysisDetail } from "@/lib/api";
import {
  UploadCloud,
  Shield,
  ShieldAlert,
  BarChart2,
  FileKey,
  List,
  AlertCircle,
  FileText,
  MessageSquare,
  Download,
  Share2,
  Database,
  Bot,
  Send,
} from "lucide-react";

type User = { id: number; email: string; role: string };
type Props = { user: User; onLogout?: () => void };

export default function AnalyzePage({ user, onLogout }: Props) {
  const [searchParams] = useSearchParams();
  const analysisIdFromUrl = searchParams.get("analysis");
  const [file, setFile] = useState<File | null>(null);
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [complianceCheck, setComplianceCheck] = useState(true);
  const [riskAssessment, setRiskAssessment] = useState(true);
  const [keyDataOcr, setKeyDataOcr] = useState(false);
  const [useMl, setUseMl] = useState(true);
  const [useLlm, setUseLlm] = useState(false);

  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const [savingToDb, setSavingToDb] = useState(false);
  const [saveToDbErr, setSaveToDbErr] = useState<string | null>(null);
  const [shareFeedback, setShareFeedback] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Document chat (LFM) — current analysis in memory
  type DocChatMessage = { role: "user" | "assistant"; content: string };
  const [documentChatMessages, setDocumentChatMessages] = useState<DocChatMessage[]>([]);
  const [documentChatInput, setDocumentChatInput] = useState("");
  const [documentChatLoading, setDocumentChatLoading] = useState(false);
  const [documentChatError, setDocumentChatError] = useState<string | null>(null);
  const documentChatScrollRef = useRef<HTMLDivElement>(null);

  const ruleHits = useMemo(() => (data?.rule_hits as Record<string, unknown>[]) ?? [], [data]);
  const filteredHits = useMemo(() => {
    if (severityFilter === "all") return ruleHits;
    return ruleHits.filter((h) => String(h?.severity) === severityFilter);
  }, [ruleHits, severityFilter]);

  const ruleIdToLlmExplanation = useMemo(() => {
    const byViolation = (data?.rag_by_violation as { rule_id?: string; llm_explanation?: string }[]) ?? [];
    const map: Record<string, string> = {};
    for (const b of byViolation) {
      const rid = b?.rule_id;
      const expl = b?.llm_explanation;
      if (rid && typeof expl === "string" && expl.trim()) map[String(rid)] = expl;
    }
    return map;
  }, [data?.rag_by_violation]);

  const highRiskHits = useMemo(
    () => ruleHits.filter((h) => h?.severity === "error"),
    [ruleHits]
  );
  const mediumRiskHits = useMemo(
    () => ruleHits.filter((h) => h?.severity === "high"),
    [ruleHits]
  );

  const handleSaveToDb = useCallback(async () => {
    if (!data) return;
    setSaveToDbErr(null);
    setSavingToDb(true);
    try {
      const result = await saveAnalysisToDb({
        filename: file?.name ?? "uploaded_contract.pdf",
        result_json: JSON.stringify(data),
        mime_type: file?.type ?? "application/pdf",
        page_count: (data?.ocr_chunks as unknown[])?.length ?? null,
      });
      setAnalysisId(result.id);
      if (typeof data === "object" && data !== null) {
        (data as Record<string, unknown>).analysis_id = result.id;
      }
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "response" in e
          ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      setSaveToDbErr(typeof msg === "string" ? msg : "Failed to save");
    } finally {
      setSavingToDb(false);
    }
  }, [data, file]);

  const exportReport = useCallback(() => {
    if (!data) return;
    const report = {
      contract_type: data.contract_type,
      law_scope_used: data.law_scope_used,
      rule_hits: data.rule_hits,
      labor_summary: data.labor_summary,
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `legato-analysis-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [data]);

  const analyze = useCallback(async () => {
    setErr(null);
    if (!file) {
      setErr("Choose a contract file first.");
      return;
    }

    setLoading(true);
    setAnalysisId(null);
    try {
      const res = await analyzeContract(file, {
        useRag: true,
        useMl,
        useLlm,
        llmTopK: 2,
        llmMaxNewTokens: 200,
        save: true,
      });
      const resObj = res as Record<string, unknown>;
      setData(resObj);
      if (typeof resObj?.analysis_id === "number") setAnalysisId(resObj.analysis_id as number);
    } catch (e: unknown) {
      const res = e && typeof e === "object" && "response" in e ? (e as { response?: { data?: unknown; status?: number } }).response : null;
      const dataDetail = res?.data && typeof res.data === "object" && res.data !== null && "detail" in res.data ? (res.data as { detail?: unknown }).detail : null;
      let msg: string;
      if (typeof dataDetail === "string") msg = dataDetail;
      else if (Array.isArray(dataDetail) && dataDetail.length) msg = dataDetail.map((x: { msg?: string }) => x?.msg ?? "").filter(Boolean).join("; ") || "Validation error";
      else msg = res?.status === 401 ? "Please log in to save analysis." : res?.status === 503 ? "Backend not fully configured (DB or services)." : "Analysis failed. Please try again or check your connection.";
      setErr(msg);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [file, useMl, useLlm]);

  useEffect(() => {
    if (!analysisIdFromUrl) return;
    const id = parseInt(analysisIdFromUrl, 10);
    if (isNaN(id)) return;
    getAnalysis(id)
      .then((a: AnalysisDetail) => {
        const parsed = typeof a.result_json === "string" ? JSON.parse(a.result_json) : a.result_json;
        setData(parsed as Record<string, unknown>);
        setAnalysisId(a.id);
      })
      .catch(() => setErr("Failed to load analysis."));
  }, [analysisIdFromUrl]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f && (f.type === "application/pdf" || f.type.includes("image") || f.name.endsWith(".docx"))) {
        setFile(f);
        setErr(null);
      }
    },
    []
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const browseFiles = () => fileInputRef.current?.click();

  // Build document context for LFM chat: OCR + violations (ML/RAG) + summaries so LFM answers accurately
  const documentContext = useMemo(() => {
    if (!data) return "";
    const parts: string[] = [];
    const ocrChunks = (data.ocr_chunks as Record<string, unknown>[]) ?? [];
    if (ocrChunks.length) {
      const ocrText = ocrChunks
        .map((c) => String(c?.normalized_text ?? c?.text ?? "").trim())
        .filter(Boolean)
        .join("\n\n");
      if (ocrText) parts.push("## Contract OCR Text\n" + ocrText.slice(0, 10000));
    }
    // Include violations (rule_hits from ML/rule engine) so LFM can answer "what are the violations?"
    const ruleHits = (data.rule_hits as Record<string, unknown>[]) ?? [];
    if (ruleHits.length) {
      const violationsText = ruleHits
        .map((h, i) => {
          const rid = h.rule_id ?? h.id ?? `violation_${i + 1}`;
          const sev = h.severity ?? "info";
          const desc = h.description ?? "";
          const matched = h.matched_text ? ` (matched: ${String(h.matched_text).slice(0, 150)}...)` : "";
          return `- [${sev}] ${rid}: ${desc}${matched}`;
        })
        .join("\n");
      parts.push("## Detected Violations (from ML/RAG analysis)\n" + violationsText.slice(0, 4000));
    }
    const labor = data.labor_summary as Record<string, unknown> | undefined;
    if (labor && typeof labor === "object")
      parts.push("## Labor Summary\n" + JSON.stringify(labor, null, 0).slice(0, 2000));
    const cb = data.cross_border_summary as Record<string, unknown> | undefined;
    if (cb && typeof cb === "object")
      parts.push("## Cross-Border Summary\n" + JSON.stringify(cb, null, 0).slice(0, 1000));
    return parts.join("\n\n") || "";
  }, [data]);

  const handleDocumentChatSend = useCallback(async () => {
    const trimmed = documentChatInput.trim();
    if (!trimmed || !documentContext || documentChatLoading) return;
    setDocumentChatError(null);
    setDocumentChatMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setDocumentChatInput("");
    setDocumentChatLoading(true);
    try {
      const res = await chatWithDocument({ document_context: documentContext, message: trimmed });
      const content = typeof res?.content === "string" ? res.content : "No reply.";
      setDocumentChatMessages((prev) => [...prev, { role: "assistant", content }]);
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "response" in e
          ? (e as { response?: { data?: { detail?: string }; status?: number } }).response?.data?.detail
          : null;
      const isTimeout =
        e && typeof e === "object" && "code" in e && (e as { code?: string }).code === "ECONNABORTED";
      const errMsg =
        typeof msg === "string"
          ? msg
          : isTimeout
            ? "Request timed out. The first message can take 1–2 minutes while the model loads. Try again."
            : "Document chat failed. Is the local LFM model available?";
      setDocumentChatError(errMsg);
      setDocumentChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: `[Error: ${errMsg}]` },
      ]);
    } finally {
      setDocumentChatLoading(false);
    }
  }, [documentChatInput, documentContext, documentChatLoading]);

  useEffect(() => {
    documentChatScrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [documentChatMessages]);

  const contractTitle = file?.name?.replace(/\.[^.]+$/, "") ?? "Contract";
  const pageCount = Array.isArray(data?.ocr_chunks) ? (data.ocr_chunks as unknown[]).length : 0;

  return (
    <AppShell
      user={user}
      onLogout={onLogout}
      title="Contract Analysis"
      subtitle="Analyze enterprise legal agreements with precision LLMs"
      right={loading ? <span className="text-sm text-muted-foreground">Analysis in progress…</span> : undefined}
    >
      <div className="grid gap-6 lg:grid-cols-5">
        {/* Left Pane: Upload & Settings */}
        <div className="lg:col-span-2 space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Contract Analysis</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Analyze enterprise legal agreements with precision LLMs
            </p>
          </div>

          {/* Drop Zone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            className="border-2 border-dashed border-muted-foreground/30 rounded-lg p-8 flex flex-col items-center justify-center gap-3 min-h-[220px] bg-muted/30 hover:bg-muted/50 transition-colors"
          >
            <UploadCloud className="h-12 w-12 text-muted-foreground" />
            <p className="font-semibold text-foreground">Drop your contract</p>
            <p className="text-sm text-muted-foreground">PDF, DOCX (up to 20MB)</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,image/*"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <Button onClick={browseFiles} className="mt-2">
              Browse Files
            </Button>
            {file && (
              <p className="text-xs text-muted-foreground truncate max-w-full px-4">
                Selected: {file.name}
              </p>
            )}
            {err && <p className="text-sm text-destructive">{err}</p>}
          </div>

          {/* Analysis Settings */}
          <Card className="bg-muted/30 border-muted">
            <CardContent className="p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-4">
                Analysis Settings
              </p>
              <div className="space-y-4">
                <label className="flex items-center justify-between gap-3 cursor-pointer p-3 rounded-lg bg-background/80 hover:bg-background transition-colors">
                  <div className="flex items-center gap-3">
                    <Shield className="h-5 w-5 text-primary" />
                    <span className="text-sm font-medium">Compliance Check</span>
                  </div>
                  <Checkbox checked={complianceCheck} onCheckedChange={(v) => setComplianceCheck(!!v)} />
                </label>
                <label className="flex items-center justify-between gap-3 cursor-pointer p-3 rounded-lg bg-background/80 hover:bg-background transition-colors">
                  <div className="flex items-center gap-3">
                    <ShieldAlert className="h-5 w-5 text-primary" />
                    <span className="text-sm font-medium">Risk Assessment</span>
                  </div>
                  <Checkbox checked={riskAssessment} onCheckedChange={(v) => setRiskAssessment(!!v)} />
                </label>
                <label className="flex items-center justify-between gap-3 cursor-pointer p-3 rounded-lg bg-background/80 hover:bg-background transition-colors">
                  <div className="flex items-center gap-3">
                    <FileKey className="h-5 w-5 text-muted-foreground" />
                    <span className="text-sm font-medium">Key Data OCR</span>
                  </div>
                  <Checkbox checked={keyDataOcr} onCheckedChange={(v) => setKeyDataOcr(!!v)} />
                </label>
                <label className="flex items-center justify-between gap-3 cursor-pointer p-3 rounded-lg bg-background/80 hover:bg-background transition-colors">
                  <div className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-3">
                      <BarChart2 className="h-5 w-5 text-primary" />
                      <span className="text-sm font-medium">ML-assisted analysis</span>
                    </div>
                    <span className="text-xs text-muted-foreground pl-8">Uses trained ML model to detect violations; results appear in the Violations tab.</span>
                  </div>
                  <Checkbox checked={useMl} onCheckedChange={(v) => setUseMl(!!v)} />
                </label>
              </div>
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 cursor-pointer p-3 rounded-lg bg-background/80 hover:bg-background transition-colors w-full justify-between">
                  <span className="text-sm font-medium">LLM explanations (LFM)</span>
                  <Checkbox checked={useLlm} onCheckedChange={(v) => setUseLlm(!!v)} />
                </label>
              </div>
            </CardContent>
          </Card>

          <Button
            className="w-full"
            size="lg"
            onClick={analyze}
            disabled={loading || !file}
          >
            {loading ? "Analyzing..." : "Analyze Contract"}
          </Button>
        </div>

        {/* Right Pane: Results */}
        <div className="lg:col-span-3">
          {!data ? (
            <div className="flex flex-col items-center justify-center min-h-[400px] text-center p-8 border rounded-lg bg-muted/20">
              <FileText className="h-16 w-16 text-muted-foreground/50 mb-4" />
              <p className="text-muted-foreground">
                Upload a contract and run analysis to see results here.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              <Tabs defaultValue="summary" className="w-full">
                <TabsList className="w-full justify-start border-b rounded-none h-auto p-0 bg-transparent">
                  <TabsTrigger
                    value="summary"
                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent"
                  >
                    <List className="h-4 w-4 mr-2" />
                    Summary
                  </TabsTrigger>
                  <TabsTrigger
                    value="violations"
                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent"
                  >
                    <AlertCircle className="h-4 w-4 mr-2" />
                    Violations
                    {ruleHits.length > 0 && (
                      <span className="ml-2 flex h-5 w-5 items-center justify-center rounded-full bg-destructive text-destructive-foreground text-xs">
                        {ruleHits.length}
                      </span>
                    )}
                  </TabsTrigger>
                  <TabsTrigger
                    value="ocr"
                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent"
                  >
                    <FileText className="h-4 w-4 mr-2" />
                    OCR
                  </TabsTrigger>
                  <TabsTrigger
                    value="rag"
                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent"
                  >
                    <MessageSquare className="h-4 w-4 mr-2" />
                    AI Chat (RAG)
                  </TabsTrigger>
                  <TabsTrigger
                    value="document-chat"
                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent"
                  >
                    <Bot className="h-4 w-4 mr-2" />
                    Chat with document (LFM)
                  </TabsTrigger>
                </TabsList>

                {/* Contract header */}
                <div className="flex items-start justify-between gap-4 py-4">
                  <div>
                    <h2 className="text-lg font-semibold">{contractTitle}</h2>
                    <p className="text-sm text-muted-foreground mt-0.5">
                      Last analyzed Today at {new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })} • PDF • {pageCount || "—"} Pages
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button variant="outline" size="icon" onClick={exportReport} title="Export report">
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-2"
                      onClick={handleSaveToDb}
                      disabled={savingToDb}
                      title="Save to database"
                    >
                      <Database className="h-4 w-4" />
                      {analysisId ? `Saved (ID: ${analysisId})` : savingToDb ? "Saving…" : "Save to DB"}
                    </Button>
                    {saveToDbErr && <span className="text-sm text-destructive">{saveToDbErr}</span>}
                    {shareFeedback && (
                      <span className="text-sm text-muted-foreground">{shareFeedback}</span>
                    )}
                    <Button
                      variant="outline"
                      size="icon"
                      title="Share"
                      onClick={async () => {
                        const url = window.location.href;
                        setShareFeedback(null);
                        try {
                          if (navigator.share) {
                            await navigator.share({
                              title: contractTitle,
                              text: "Legato contract analysis report",
                              url,
                            });
                            setShareFeedback("Shared!");
                          } else {
                            await navigator.clipboard.writeText(url);
                            setShareFeedback("Link copied!");
                          }
                        } catch {
                          try {
                            await navigator.clipboard.writeText(url);
                            setShareFeedback("Link copied!");
                          } catch {
                            setShareFeedback("Share failed");
                          }
                        }
                        setTimeout(() => setShareFeedback(null), 2000);
                      }}
                    >
                      <Share2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                <TabsContent value="summary" className="mt-0 space-y-6">
                  <div>
                    <h3 className="text-sm font-bold uppercase text-primary mb-2">
                      Executive Summary
                    </h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      This {String(data?.contract_type ?? "employment")} agreement has been analyzed
                      for compliance with {(data?.law_scope_used as string[])?.join(", ") || "labor"} law.
                      {ruleHits.length > 0
                        ? ` ${ruleHits.length} potential issue(s) were flagged for review.`
                        : " No critical violations were detected."}
                    </p>
                  </div>

                  {/* Risk Flags */}
                  {(highRiskHits.length > 0 || mediumRiskHits.length > 0) && (
                    <div className="grid gap-4 sm:grid-cols-2">
                      {highRiskHits.length > 0 && (
                        <Card className="border-l-4 border-l-destructive bg-destructive/5">
                          <CardContent className="p-4">
                            <div className="flex items-start gap-3">
                              <AlertCircle className="h-6 w-6 text-destructive shrink-0 mt-0.5" />
                              <div>
                                <p className="text-xs font-bold uppercase text-destructive mb-1">
                                  High Risk Flag
                                </p>
                                <p className="font-semibold text-foreground">
                                  {(highRiskHits[0] as Record<string, unknown>).rule_id as string}
                                </p>
                                <p className="text-sm text-muted-foreground mt-1">
                                  {(highRiskHits[0] as Record<string, unknown>).description as string}
                                </p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                      {mediumRiskHits.length > 0 && (
                        <Card className="border-l-4 border-l-orange-500 bg-orange-500/5">
                          <CardContent className="p-4">
                            <div className="flex items-start gap-3">
                              <ShieldAlert className="h-6 w-6 text-orange-500 shrink-0 mt-0.5" />
                              <div>
                                <p className="text-xs font-bold uppercase text-orange-600 mb-1">
                                  Medium Risk Flag
                                </p>
                                <p className="font-semibold text-foreground">
                                  {(mediumRiskHits[0] as Record<string, unknown>).rule_id as string}
                                </p>
                                <p className="text-sm text-muted-foreground mt-1">
                                  {(mediumRiskHits[0] as Record<string, unknown>).description as string}
                                </p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  )}

                  {/* Key Metadata */}
                  <div>
                    <h3 className="text-sm font-bold uppercase text-primary mb-3">
                      Key Metadata
                    </h3>
                    <div className="rounded-lg border overflow-hidden">
                      <table className="w-full text-sm">
                        <tbody>
                          <tr className="border-b">
                            <td className="px-4 py-3 font-medium text-muted-foreground w-1/3">Contract Type</td>
                            <td className="px-4 py-3">{String(data?.contract_type ?? "—")}</td>
                          </tr>
                          <tr className="border-b">
                            <td className="px-4 py-3 font-medium text-muted-foreground">Law Scope</td>
                            <td className="px-4 py-3">
                              {(data?.law_scope_used as string[])?.join(", ") ?? "—"}
                            </td>
                          </tr>
                          <tr>
                            <td className="px-4 py-3 font-medium text-muted-foreground">Cross-Border</td>
                            <td className="px-4 py-3">
                              {String((data?.cross_border_summary as Record<string, unknown>)?.status ?? "—")}
                            </td>
                          </tr>
                          <tr>
                            <td className="px-4 py-3 font-medium text-muted-foreground">ML-assisted analysis</td>
                            <td className="px-4 py-3">
                              {data?.ml_used === true ? "Yes" : data?.ml_used === false ? "No" : "—"}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="violations" className="mt-0">
                  <div className="flex items-center gap-2 mb-4">
                    <select
                      className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                      value={severityFilter}
                      onChange={(e) => setSeverityFilter(e.target.value)}
                    >
                      <option value="all">All severities</option>
                      <option value="error">Error</option>
                      <option value="high">High</option>
                      <option value="info">Info</option>
                    </select>
                    <Badge variant="outline">{filteredHits.length} violation{filteredHits.length !== 1 ? "s" : ""}</Badge>
                  </div>
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-3">
                      {filteredHits.map((h, idx) => {
                        const ruleId = String(h.rule_id ?? h.id ?? "");
                        const matchedText = h.matched_text != null ? String(h.matched_text).trim() : "";
                        return (
                          <Card key={ruleId ? `${ruleId}-${String(h.chunk_id ?? "")}-${idx}` : idx}>
                            <CardContent className="p-4">
                              <div className="flex items-center justify-between gap-3">
                                <span className="font-medium">{ruleId || "Violation"}</span>
                                <Badge
                                  variant={
                                    h.severity === "error"
                                      ? "destructive"
                                      : h.severity === "high"
                                      ? "default"
                                      : "secondary"
                                  }
                                  className={
                                    h.severity === "error"
                                      ? "bg-destructive text-destructive-foreground"
                                      : h.severity === "high"
                                      ? "bg-amber-500/90 text-white border-0"
                                      : "bg-muted text-muted-foreground"
                                  }
                                >
                                  {h.severity === "error"
                                    ? "Error"
                                    : h.severity === "high"
                                    ? "High"
                                    : "Info"}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground mt-2">{String(h.description ?? "")}</p>
                              {matchedText ? (
                                <div className="mt-2 p-2 rounded-md bg-muted/60 border border-border">
                                  <p className="text-xs font-medium text-muted-foreground mb-1">Matched contract text</p>
                                  <p className="text-sm whitespace-pre-wrap break-words">{matchedText}</p>
                                </div>
                              ) : null}
                              <p className="text-xs text-muted-foreground mt-2">
                                Article: {String(h.article ?? "—")} | Chunk: {String(h.chunk_id ?? "—")}
                              </p>
                              {ruleIdToLlmExplanation[ruleId] && (
                                <div className="mt-3 pt-3 border-t border-border">
                                  <p className="text-xs font-medium text-primary mb-1">LLM explanation (LFM)</p>
                                  <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                                    {ruleIdToLlmExplanation[ruleId]}
                                  </p>
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  </ScrollArea>
                </TabsContent>

                <TabsContent value="ocr" className="mt-0">
                  <ScrollArea className="h-[500px]">
                    <div className="space-y-4">
                      {((data?.ocr_chunks as Record<string, unknown>[]) ?? []).map((c) => (
                        <Card key={String(c.id)}>
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-medium">Page {Number(c.page ?? 0) + 1}</span>
                              <Badge variant="outline">{String(c.id ?? "")}</Badge>
                            </div>
                            <p className="text-sm whitespace-pre-wrap leading-relaxed text-muted-foreground">
                              {String(c.normalized_text ?? c.text ?? "")}
                            </p>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </TabsContent>

                <TabsContent value="rag" className="mt-0">
                  <ScrollArea className="h-[500px]">
                    <div className="space-y-4">
                      {((data?.rag_legal_hits as Record<string, unknown>[]) ?? []).map((r, idx) => (
                        <Card key={idx}>
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-medium">
                                Article {String((r?.metadata as Record<string, unknown>)?.article ?? "—")}
                              </span>
                              <Badge variant="outline">
                                {Number(r?.score ?? 0).toFixed(3)}
                              </Badge>
                            </div>
                            <p className="text-sm whitespace-pre-wrap leading-relaxed text-muted-foreground">
                              {String(r?.text ?? "")}
                            </p>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </TabsContent>

                <TabsContent value="document-chat" className="mt-0">
                  <div className="flex flex-col rounded-lg border h-[500px]">
                    <p className="text-xs text-muted-foreground px-4 py-2 border-b">
                      Chat with this document using the local LFM model. Ask questions about the contract. First message may take 1–2 minutes while the model loads.
                    </p>
                    <ScrollArea className="flex-1 p-4">
                      <div className="space-y-4">
                        {documentChatMessages.length === 0 && (
                          <p className="text-sm text-muted-foreground">No messages yet. Type a question below.</p>
                        )}
                        {documentChatMessages.map((m, idx) => (
                          <div
                            key={idx}
                            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                          >
                            <div
                              className={`max-w-[85%] rounded-lg px-4 py-2 text-sm ${
                                m.role === "user"
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-foreground"
                              }`}
                            >
                              <p className="whitespace-pre-wrap">{m.content}</p>
                            </div>
                          </div>
                        ))}
                        {documentChatLoading && (
                          <div className="flex justify-start">
                            <div className="rounded-lg px-4 py-2 text-sm bg-muted text-muted-foreground">
                              Thinking…
                            </div>
                          </div>
                        )}
                        <div ref={documentChatScrollRef} />
                      </div>
                    </ScrollArea>
                    {documentChatError && (
                      <p className="text-sm text-destructive px-4 pb-2">{documentChatError}</p>
                    )}
                    <div className="flex gap-2 p-4 border-t">
                      <input
                        type="text"
                        className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                        placeholder="Ask about this contract…"
                        value={documentChatInput}
                        onChange={(e) => setDocumentChatInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleDocumentChatSend()}
                        disabled={documentChatLoading || !documentContext}
                      />
                      <Button
                        size="icon"
                        onClick={handleDocumentChatSend}
                        disabled={documentChatLoading || !documentChatInput.trim() || !documentContext}
                        title="Send"
                      >
                        <Send className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
