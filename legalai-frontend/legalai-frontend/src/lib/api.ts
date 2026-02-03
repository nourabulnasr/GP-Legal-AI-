import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000 ,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

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

  const { data } = await api.post("/ocr_check_and_search", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 180000,
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
export async function adminListAll(): Promise<AnalysisDetail[]> {
  const { data } = await api.get<AnalysisDetail[]>("/analyses/admin/all");
  return data;
}

export async function adminListUserAnalyses(userId: number): Promise<AnalysisItem[]> {
  const { data } = await api.get<AnalysisItem[]>(`/analyses/admin/user/${userId}`);
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

// ---- Chat with document (LFM) ----
export async function chatWithDocument(payload: {
  document_context: string;
  message: string;
  history?: { role: string; content: string }[];
}): Promise<{ content: string }> {
  const { data } = await api.post<{ content: string }>("/chat/document", payload, { timeout: 120000 });
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
