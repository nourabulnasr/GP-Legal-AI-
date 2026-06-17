import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status;
    // If token is expired/invalid, clear it so the app can recover cleanly.
    if (status === 401) {
      localStorage.removeItem("access_token");
    }
    return Promise.reject(error);
  }
);

// ---- OCR + Check endpoint ----
export async function analyzeContract(
  file: File,
  options?: {
    useRag?: boolean;
    useMl?: boolean;
    useLlm?: boolean;
    llmTopK?: number;
    llmMaxNewTokens?: number;
    save?: boolean;
    query?: string;
    translateToAr?: boolean;
    /** When true, each OCR page uses its own detected language for MT (best for mixed-language PDFs). */
    translatePerChunkMt?: boolean;
    /** OCR + detect + optional MT only; skips Egyptian labor rules, ML, RAG, LLM */
    translationOnly?: boolean;
    sourceLanguageMode?: "auto" | "manual";
    sourceLanguageOverride?: string;
    translationTargetLang?: "ar" | "en" | "fr" | "de";
  }
) {
  const form = new FormData();
  form.append("file", file);
  form.append("use_rag", String(options?.useRag ?? true));
  form.append("use_ml", String(options?.useMl ?? true));
  form.append("use_llm", String(options?.useLlm ?? false));
  if (options?.llmTopK != null) form.append("llm_top_k", String(options.llmTopK));
  if (options?.llmMaxNewTokens != null) form.append("llm_max_new_tokens", String(options.llmMaxNewTokens));
  form.append("save", String(options?.save ?? true));
  if (options?.query) form.append("query", options.query);
  form.append("translate_to_ar", String(options?.translateToAr ?? false));
  form.append("translate_per_chunk_mt", String(options?.translatePerChunkMt ?? false));
  form.append("translation_only", String(options?.translationOnly ?? false));
  form.append("source_language_mode", String(options?.sourceLanguageMode ?? "auto"));
  form.append("translation_target_lang", String(options?.translationTargetLang ?? "ar"));
  if (options?.sourceLanguageMode === "manual" && options?.sourceLanguageOverride?.trim())
    form.append("source_language_override", options.sourceLanguageOverride.trim());

  const { data } = await api.post("/ocr_check_and_search", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 600000,
  });
  return data;
}

// ---- Analyses (user history) ----
export type AnalysisItem = { id: number; filename: string | null; created_at: string };
export type AnalysisDetail = AnalysisItem & { user_id: number; result_json: string };

export async function listAnalyses(): Promise<AnalysisItem[]> {
  const { data } = await api.get<AnalysisItem[]>("/analyses");
  return data;
}

export async function getAnalysis(id: number): Promise<AnalysisDetail> {
  const { data } = await api.get<AnalysisDetail>(`/analyses/${id}`);
  return data;
}

export async function deleteAnalysis(id: number): Promise<{ status: string; deleted_id: number }> {
  const { data } = await api.delete<{ status: string; deleted_id: number }>(`/analyses/${id}`);
  return data;
}

// ---- Admin ----
export type AdminUser = { id: number; email: string; role: string };

export async function adminListAll(): Promise<AnalysisDetail[]> {
  const { data } = await api.get<AnalysisDetail[]>("/analyses/admin/all");
  return data;
}

export async function adminListUsers(): Promise<AdminUser[]> {
  const { data } = await api.get<AdminUser[]>("/analyses/admin/users");
  return data;
}

export async function adminUpdateUserRole(userId: number, role: "admin" | "user"): Promise<AdminUser> {
  const { data } = await api.patch<AdminUser>(`/analyses/admin/users/${userId}`, { role });
  return data;
}

export async function adminDeleteUser(userId: number): Promise<{ status: string; deleted_id: number; email: string }> {
  const { data } = await api.delete<{ status: string; deleted_id: number; email: string }>(
    `/analyses/admin/users/${userId}`,
  );
  return data;
}

export async function adminListUserAnalyses(userId: number): Promise<AnalysisItem[]> {
  const { data } = await api.get<AnalysisItem[]>(`/analyses/admin/user/${userId}`);
  return data;
}

// ---- Admin: labor law + RAG rebuild ----
export type LawJobResponse = {
  job_id: string;
  status: string;
  saved_pdf?: string;
  result?: unknown;
  error?: string;
  mode?: string;
};

export async function adminLawPreviewPdf(
  file: File,
  lawDisplayName?: string
): Promise<{ article_count: number; preview: { article: string; title: string; text_len: number }[] }> {
  const form = new FormData();
  form.append("file", file);
  if (lawDisplayName?.trim()) form.append("law_display_name", lawDisplayName.trim());
  const { data } = await api.post("/admin/law/preview-pdf", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 120000,
  });
  return data;
}

export async function adminLawUploadPdf(file: File, lawDisplayName?: string): Promise<LawJobResponse> {
  const form = new FormData();
  form.append("file", file);
  if (lawDisplayName?.trim()) form.append("law_display_name", lawDisplayName.trim());
  const { data } = await api.post<LawJobResponse>("/admin/law/upload-pdf", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 120000,
  });
  return data;
}

export async function adminLawReindex(): Promise<LawJobResponse> {
  const { data } = await api.post<LawJobResponse>("/admin/law/reindex", {}, { timeout: 120000 });
  return data;
}

export async function adminLawJob(jobId: string): Promise<LawJobResponse> {
  const { data } = await api.get<LawJobResponse>(`/admin/law/jobs/${jobId}`);
  return data;
}

/** Cleaned chunk JSONL (same artifact used for Chroma ingest). Requires admin; 404 if file missing. */
export async function adminLawDownloadChunksCleaned(): Promise<Blob> {
  const { data } = await api.get<Blob>("/admin/law/download-chunks-cleaned", {
    responseType: "blob",
    timeout: 120000,
  });
  return data;
}

export async function adminLawSyncFromUrl(): Promise<Record<string, unknown>> {
  const { data } = await api.post<Record<string, unknown>>("/admin/law/sync-from-url", {}, { timeout: 300000 });
  return data;
}

// ---- Chat (Gemini) ----
export async function chatMessage(payload: {
  analysis_id: number;
  message: string;
  history?: { role: string; content: string }[];
}): Promise<{ content: string; analysis_id: number }> {
  const { data } = await api.post("/chat/message", payload);
  return data;
}

// ---- General assistant chat (no contract context) ----
export async function chatAssistant(payload: {
  message: string;
  history?: { role: string; content: string }[];
}): Promise<{ content: string }> {
  const { data } = await api.post<{ content: string }>("/chat/assistant", payload);
  return data;
}

// ---- Chat with document (LFM) ----
export async function chatWithDocument(payload: {
  document_context?: string;
  analysis_id?: number;
  message: string;
  history?: { role: string; content: string }[];
}): Promise<{ content: string }> {
  const { data } = await api.post<{ content: string }>("/chat/document", payload, { timeout: 300000 });
  return data;
}

// ---- Save analysis to DB ----
export async function saveAnalysisToDb(payload: {
  filename: string;
  result_json: string;
  mime_type?: string;
  sha256?: string;
  page_count?: number;
  ocr_used?: number;
  detected_lang?: string;
}): Promise<{ id: number; filename: string; created_at: string }> {
  const { data } = await api.post("/analyses", payload);
  return data;
}
