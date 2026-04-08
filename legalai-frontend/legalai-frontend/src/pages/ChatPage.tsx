import { useState, useRef, useEffect } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { logout } from "@/lib/auth";
import { listAnalyses, chatAssistant, chatWithDocument, getAnalysis } from "@/lib/api";
import type { AnalysisDetail } from "@/lib/api";
import {
  Scale,
  Bot,
  User,
  Send,
  LogOut,
  MessageCircle,
  X,
  FileText,
} from "lucide-react";

type Message = {
  id: string;
  role: "assistant" | "user";
  content: string;
};

type Props = {
  user: { id: number; email: string; role: string };
  onLogout?: () => void;
};

const CHAT_QUOTA_MESSAGE =
  "The AI chat has reached its usage limit for now. Please try again in a few minutes, or check your API plan and billing.";

export default function ChatPage({ user, onLogout }: Props) {
  const [searchParams] = useSearchParams();
  const analysisIdFromUrl = searchParams.get("analysis");
  const [analyses, setAnalyses] = useState<{ id: number; filename: string | null; created_at: string }[]>([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<number | null>(null);
  const [analysisDetail, setAnalysisDetail] = useState<AnalysisDetail | null>(null);
  const [chatMode, setChatMode] = useState<"general" | "contract">("general");
  const [chatOpen, setChatOpen] = useState(false);
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
    if (!selectedAnalysisId) {
      setAnalysisDetail(null);
      return;
    }
    getAnalysis(selectedAnalysisId)
      .then(setAnalysisDetail)
      .catch(() => setAnalysisDetail(null));
  }, [selectedAnalysisId]);

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
    if (chatMode === "contract" && !selectedAnalysisId) {
      setChatError("Select an analysis to use contract context.");
      return;
    }

    setChatError(null);
    setInput("");
    const userMsg: Message = { id: Date.now().toString(), role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    const history = messages.map((m) => ({ role: m.role, content: m.content }));

    try {
      if (chatMode === "general") {
        const res = await chatAssistant({ message: trimmed, history });
        setMessages((prev) => [
          ...prev,
          { id: (Date.now() + 1).toString(), role: "assistant", content: res.content },
        ]);
      } else {
        const res = await chatWithDocument({
          analysis_id: selectedAnalysisId!,
          message: trimmed,
          history,
        });
        setMessages((prev) => [
          ...prev,
          { id: (Date.now() + 1).toString(), role: "assistant", content: res.content },
        ]);
      }
    } catch (e: unknown) {
      const err = e as { response?: { status?: number; data?: { detail?: string } } };
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail;
      let content: string;
      if (status === 429 || (typeof detail === "string" && (detail.includes("429") || detail.toLowerCase().includes("quota") || detail.toLowerCase().includes("rate limit")))) {
        content = CHAT_QUOTA_MESSAGE;
      } else {
        content = typeof detail === "string" ? detail : "Chat failed. For contract mode, ensure the local LFM model is available.";
      }
      setMessages((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: "assistant", content },
      ]);
    } finally {
      setIsTyping(false);
    }
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
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top bar */}
      <header className="border-b bg-card px-4 py-3 flex items-center justify-between shrink-0">
        <Link to="/analyze" className="flex items-center gap-2">
          <Scale className="h-6 w-6 text-primary" />
          <span className="font-bold text-foreground">Legato</span>
        </Link>
        <div className="flex items-center gap-3">
          <Link to="/analyze">
            <Button variant="ghost" size="sm">Analyze</Button>
          </Link>
          <Link to="/history">
            <Button variant="ghost" size="sm">History</Button>
          </Link>
          <div className="flex items-center gap-2 pl-2 border-l">
            <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center text-sm font-semibold text-primary shrink-0">
              {initials}
            </div>
            <span className="text-sm text-muted-foreground hidden sm:inline">{user.email.split("@")[0]}</span>
            <Button variant="ghost" size="sm" className="gap-2" onClick={handleLogout}>
              <LogOut className="h-4 w-4" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center justify-center p-6 text-center">
        <FileText className="h-16 w-16 text-muted-foreground/60 mx-auto mb-4" />
        <h1 className="text-2xl font-bold text-foreground mb-2">Chat with Legato</h1>
        <p className="text-muted-foreground max-w-md mb-6">
          Ask general legal questions or select an analysis to chat about a specific contract.
        </p>
        <Button
          size="lg"
          className="gap-2"
          onClick={() => setChatOpen(true)}
        >
          <MessageCircle className="h-5 w-5" />
          Open AI Assistant
        </Button>
      </main>

      {/* Floating chat button (bottom-right) */}
      {!chatOpen && (
        <button
          type="button"
          onClick={() => setChatOpen(true)}
          className="fixed bottom-6 right-6 h-14 w-14 rounded-full bg-primary text-primary-foreground shadow-lg flex items-center justify-center hover:opacity-90 transition-opacity z-50"
          aria-label="Open AI chat"
        >
          <Bot className="h-7 w-7" />
        </button>
      )}

      {/* Chat panel (slide-up / overlay) */}
      {chatOpen && (
        <div className="fixed inset-0 z-50 flex flex-col sm:inset-auto sm:bottom-6 sm:right-6 sm:top-auto sm:left-auto sm:w-[420px] sm:max-h-[calc(100vh-8rem)] sm:rounded-xl sm:shadow-2xl border bg-card flex flex-col overflow-hidden">
          {/* Panel header */}
          <div className="flex items-center justify-between px-4 py-3 border-b shrink-0">
            <div className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-primary" />
              <span className="font-semibold">Legato Assistant</span>
            </div>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setChatOpen(false)} aria-label="Close chat">
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Mode + analysis (for contract) */}
          <div className="px-4 py-2 border-b space-y-2 shrink-0">
            <div className="flex gap-2">
              <Button
                variant={chatMode === "general" ? "default" : "outline"}
                size="sm"
                onClick={() => { setChatMode("general"); setChatError(null); }}
              >
                General
              </Button>
              <Button
                variant={chatMode === "contract" ? "default" : "outline"}
                size="sm"
                onClick={() => { setChatMode("contract"); setChatError(null); }}
              >
                Contract (LLM)
              </Button>
            </div>
            {chatMode === "contract" && (
              <select
                value={selectedAnalysisId ?? ""}
                onChange={(e) => {
                  const v = e.target.value ? parseInt(e.target.value, 10) : null;
                  setSelectedAnalysisId(v);
                  if (v) navigate(`/chat?analysis=${v}`, { replace: true });
                }}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="">Select analysis for context…</option>
                {analyses.map((a) => (
                  <option key={a.id} value={a.id}>
                    #{a.id} {a.filename ?? `Analysis ${a.id}`}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-auto p-4 space-y-4 min-h-[200px]">
            {messages.length === 0 && !isTyping && (
              <p className="text-sm text-muted-foreground text-center py-4">
                {chatMode === "general"
                  ? "Ask anything about contracts or legal topics."
                  : "Select an analysis above, then ask about that contract."}
              </p>
            )}
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-2 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
              >
                <div
                  className={`h-8 w-8 rounded-full flex items-center justify-center shrink-0 ${
                    msg.role === "assistant" ? "bg-primary/20 text-primary" : "bg-primary text-primary-foreground"
                  }`}
                >
                  {msg.role === "assistant" ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
                </div>
                <div
                  className={`rounded-lg px-3 py-2 text-sm max-w-[85%] ${
                    msg.role === "assistant" ? "bg-muted/80 text-foreground" : "bg-primary text-primary-foreground"
                  }`}
                >
                  <p className={msg.role === "user" ? "text-primary-foreground" : ""}>
                    {renderContent(msg.content)}
                  </p>
                  {msg.role === "assistant" && msg.content.includes("usage limit") && (
                    <p className="text-xs text-muted-foreground mt-1 italic">Try again in a few minutes.</p>
                  )}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex gap-2">
                <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center shrink-0">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="rounded-lg px-3 py-2 bg-muted/80">
                  <span className="inline-flex gap-1">
                    <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.3s]" />
                    <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.15s]" />
                    <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-bounce" />
                  </span>
                </div>
              </div>
            )}
            <div ref={scrollRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t shrink-0">
            {chatError && (
              <p className="text-sm text-destructive mb-2">{chatError}</p>
            )}
            <div className="flex gap-2">
              <input
                type="text"
                placeholder={chatMode === "general" ? "Ask anything..." : "Ask about this contract..."}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                className="flex-1 h-10 px-3 rounded-lg border border-input bg-background text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
              <Button
                size="icon"
                className="h-10 w-10 shrink-0"
                onClick={handleSend}
                disabled={!input.trim() || isTyping || (chatMode === "contract" && !selectedAnalysisId)}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
