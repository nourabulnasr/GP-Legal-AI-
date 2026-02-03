import { useState, useRef, useEffect } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { logout } from "@/lib/auth";
import { listAnalyses, chatMessage } from "@/lib/api";
import {
  Scale,
  FileText,
  BarChart2,
  Calendar,
  Users,
  Settings,
  Search,
  Download,
  Bot,
  User,
  Paperclip,
  Send,
  LogOut,
  Link2,
} from "lucide-react";

type Message = {
  id: string;
  role: "assistant" | "user";
  content: string;
  suggestedActions?: string[];
  quote?: string;
  keyFindings?: { label: string; value: string }[];
  sourceRef?: string;
};

type Props = {
  user: { id: number; email: string; role: string };
  onLogout?: () => void;
};

const SUGGESTED_ACTIONS = [
  "Summarize liabilities",
  "Check non-compete duration",
  "Identify governing law",
];

const CONTRACT_CONTEXT = {
  parties: "TechCorp vs. Jane Doe",
  effectiveDate: "Jan 12, 2024",
  type: "Employment Contract",
  documentName: "Employment_Agreement_v2.pdf",
};

const NAV_ITEMS = [
  { id: "document", label: "Document Info", icon: FileText, active: true },
  { id: "clauses", label: "Clause Analysis", icon: BarChart2 },
  { id: "dates", label: "Key Dates", icon: Calendar },
  { id: "signatories", label: "Signatories", icon: Users },
];

export default function ChatPage({ user, onLogout }: Props) {
  const [searchParams] = useSearchParams();
  const analysisIdFromUrl = searchParams.get("analysis");
  const [analyses, setAnalyses] = useState<{ id: number; filename: string | null; created_at: string }[]>([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    listAnalyses()
      .then(setAnalyses)
      .catch(() => setAnalyses([]));
  }, []);

  useEffect(() => {
    const id = analysisIdFromUrl ? parseInt(analysisIdFromUrl, 10) : null;
    if (id && !isNaN(id)) setSelectedAnalysisId(id);
  }, [analysisIdFromUrl]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleLogout = () => {
    logout();
    onLogout?.();
    navigate("/login", { replace: true });
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    if (!selectedAnalysisId) {
      setChatError("Select an analysis to chat.");
      return;
    }

    setChatError(null);
    setInput("");
    const userMsg: Message = { id: Date.now().toString(), role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const res = await chatMessage({
        analysis_id: selectedAnalysisId,
        message: trimmed,
        history,
      });
      setMessages((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: "assistant", content: res.content },
      ]);
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "response" in e
          ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null;
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: typeof msg === "string" ? msg : "Chat failed. Is GEMINI_API_KEY configured?",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSuggestedAction = (action: string) => {
    setInput(action);
    setChatError(null);
  };

  const renderContent = (content: string) => {
    const parts = content.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) =>
      part.startsWith("**") && part.endsWith("**") ? (
        <strong key={i}>{part.slice(2, -2)}</strong>
      ) : (
        part
      )
    );
  };

  const initials = user.email.slice(0, 2).toUpperCase();

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left sidebar - Contract Context */}
      <aside className="w-64 border-r flex flex-col bg-card shrink-0">
        <div className="p-4 border-b">
          <Link to="/analyze" className="flex items-center gap-2">
            <Scale className="h-6 w-6 text-primary" />
            <span className="font-bold text-foreground">Legato</span>
          </Link>
        </div>

        <div className="p-4 border-b">
          <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-2">
            Contract Context
          </h3>
          <select
            value={selectedAnalysisId ?? ""}
            onChange={(e) => {
              const v = e.target.value ? parseInt(e.target.value, 10) : null;
              setSelectedAnalysisId(v);
              if (v) navigate(`/chat?analysis=${v}`, { replace: true });
            }}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            <option value="">Select analysis…</option>
            {analyses.map((a) => (
              <option key={a.id} value={a.id}>
                #{a.id} {a.filename ?? `Analysis ${a.id}`}
              </option>
            ))}
          </select>
          {selectedAnalysisId ? (
            <p className="text-sm text-foreground mt-2">Analysis #{selectedAnalysisId} selected</p>
          ) : (
            <p className="text-sm text-muted-foreground mt-2">Select an analysis to chat</p>
          )}
        </div>

        <nav className="flex-1 p-3 space-y-0.5">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  item.active
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t">
          <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-3">
            Core Details
          </h3>
          <dl className="space-y-2 text-sm">
            <div>
              <dt className="text-muted-foreground font-medium">Parties</dt>
              <dd className="text-foreground">{CONTRACT_CONTEXT.parties}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground font-medium">Effective Date</dt>
              <dd className="text-foreground">{CONTRACT_CONTEXT.effectiveDate}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground font-medium">Type</dt>
              <dd className="text-foreground">{CONTRACT_CONTEXT.type}</dd>
            </div>
          </dl>
        </div>

        <div className="p-3 border-t">
          <button
            className="w-full flex items-center gap-3 px-3 py-2 text-sm text-muted-foreground hover:text-foreground rounded-lg"
            aria-label="Settings"
          >
            <Settings className="h-4 w-4" />
            Settings
          </button>
          <div className="flex items-center gap-3 px-3 py-2 mt-2">
            <div className="h-9 w-9 rounded-full bg-primary/20 flex items-center justify-center text-sm font-semibold text-primary shrink-0">
              {initials}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">
                {user.email.split("@")[0]}
              </p>
              <p className="text-xs text-muted-foreground capitalize">{user.role}</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-2 text-muted-foreground hover:text-destructive mt-1"
            onClick={handleLogout}
          >
            <LogOut className="h-4 w-4" />
            Logout
          </Button>
        </div>
      </aside>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top header */}
        <header className="border-b bg-background px-6 py-3 flex items-center justify-between gap-4 shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">
              Projects / {CONTRACT_CONTEXT.documentName}
            </span>
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-500/20 text-emerald-700">
              Active
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="search"
                placeholder="Search document..."
                className="h-9 w-64 pl-9 pr-4 rounded-md border border-input bg-muted/50 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <Button size="sm" className="gap-2">
              <Download className="h-4 w-4" />
              Export Report
            </Button>
          </div>
        </header>

        {/* Chat messages */}
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 ${
                  msg.role === "user" ? "flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`h-9 w-9 rounded-full flex items-center justify-center shrink-0 ${
                    msg.role === "assistant"
                      ? "bg-primary/20 text-primary"
                      : "bg-primary text-primary-foreground"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <Bot className="h-4 w-4" />
                  ) : (
                    <User className="h-4 w-4" />
                  )}
                </div>
                <div
                  className={`flex-1 min-w-0 ${
                    msg.role === "user" ? "flex flex-col items-end" : ""
                  }`}
                >
                  <p
                    className={`text-xs font-medium uppercase mb-1 ${
                      msg.role === "assistant"
                        ? "text-primary"
                        : "text-muted-foreground"
                    }`}
                  >
                    {msg.role === "assistant" ? "Legato Assistant" : "You"}
                  </p>
                  <div
                    className={`rounded-lg px-4 py-3 ${
                      msg.role === "assistant"
                        ? "bg-muted/80 text-foreground"
                        : "bg-primary text-primary-foreground"
                    }`}
                  >
                    <p
                      className={`text-sm leading-relaxed ${
                        msg.role === "user" ? "text-primary-foreground" : ""
                      }`}
                    >
                      {renderContent(msg.content)}
                    </p>
                    {msg.quote && (
                      <blockquote className="mt-3 pl-4 border-l-4 border-primary text-sm text-muted-foreground italic">
                        {msg.quote}
                      </blockquote>
                    )}
                    {msg.keyFindings && msg.keyFindings.length > 0 && (
                      <ul className="mt-3 space-y-1 text-sm">
                        <span className="font-medium">Key findings:</span>
                        {msg.keyFindings.map((f, i) => (
                          <li key={i}>
                            <strong>{f.label}:</strong> {f.value}
                          </li>
                        ))}
                      </ul>
                    )}
                    {msg.sourceRef && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-3 gap-2 h-8"
                        onClick={() => alert(`Sources: ${msg.sourceRef} - Link to document section will be added.`)}
                      >
                        <Link2 className="h-3 w-3" />
                        Sources: {msg.sourceRef}
                      </Button>
                    )}
                    {msg.suggestedActions && msg.suggestedActions.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-3">
                        {msg.suggestedActions.map((action) => (
                          <Button
                            key={action}
                            variant="secondary"
                            size="sm"
                            className="h-8"
                            onClick={() => handleSuggestedAction(action)}
                          >
                            {action}
                          </Button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="flex gap-3">
                <div className="h-9 w-9 rounded-full bg-primary/20 flex items-center justify-center shrink-0">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <p className="text-xs font-medium uppercase text-primary mb-1">
                    Legato Assistant
                  </p>
                  <div className="rounded-lg px-4 py-3 bg-muted/80">
                    <span className="inline-flex gap-1">
                      <span className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.3s]" />
                      <span className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.15s]" />
                      <span className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce" />
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={scrollRef} />
          </div>
        </div>

        {/* Suggested actions bar */}
        <div className="px-6 py-3 border-t">
          <div className="max-w-3xl mx-auto flex flex-wrap gap-2">
            {SUGGESTED_ACTIONS.map((action) => (
              <Button
                key={action}
                variant="outline"
                size="sm"
                className="h-8"
                onClick={() => handleSuggestedAction(action)}
              >
                {action}
              </Button>
            ))}
          </div>
        </div>

        {/* Chat input */}
        <div className="p-6 pt-0">
          {chatError && (
            <p className="max-w-3xl mx-auto mb-2 text-sm text-destructive">{chatError}</p>
          )}
          <div className="max-w-3xl mx-auto flex gap-3">
            <Button
              variant="ghost"
              size="icon"
              className="h-10 w-10 shrink-0"
              onClick={() => document.getElementById("chat-file-input")?.click()}
              title="Attach file (coming soon)"
            >
              <Paperclip className="h-4 w-4" />
            </Button>
            <input
              id="chat-file-input"
              type="file"
              className="hidden"
              accept=".pdf,.docx,image/*"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) alert(`Attachment "${f.name}" - file upload will be integrated with LLM.`);
                e.target.value = "";
              }}
            />
            <input
              type="text"
              placeholder="Ask about specific clauses, risks, or definitions..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              className="flex-1 h-10 px-4 rounded-lg border border-input bg-background text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <Button
              size="icon"
              className="h-10 w-10 shrink-0"
              onClick={handleSend}
              disabled={!input.trim() || isTyping}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-center text-xs text-muted-foreground mt-3">
            AI-GENERATED ANALYSIS. VERIFY WITH LEGAL COUNSEL.
          </p>
        </div>
      </div>
    </div>
  );
}
