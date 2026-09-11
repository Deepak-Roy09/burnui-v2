export type Role = "INSPECTOR" | "ENGINEER" | "ADMIN";
export type RiskStatus = "Normal" | "Watchlist" | "High Risk";

export interface AuthUser { id: number; full_name: string; email: string; role: Role; department: string | null; is_active: boolean; created_at: string; }
export interface TopFeature { feature: string; value: number; z_contribution: number; }
export interface ScreenedComponent { engine_id: number; risk_score: number; status: RiskStatus; z_score: number; isolation_score: number; top_features: TopFeature[]; pca_1?: number; pca_2?: number; }
export interface ScreeningParameters { sensitivity: "Standard" | "Sensitive" | "Conservative"; watchlistThreshold: number; highRiskThreshold: number; }
export interface ScreeningResponse {
  components: ScreenedComponent[];
  summary: { total_components_screened: number; normal_count: number; watchlist_count: number; high_risk_count: number; };
  screening_parameters?: { watchlist_threshold: number; high_risk_threshold: number; sensitivity: string; validated_defaults: { watchlist_threshold: number; high_risk_threshold: number; }; };
  source?: { format: "normalized_csv" | "raw_cmapss_txt"; filename: string; dataset: "FD001" | "FD002" | "FD003" | "FD004" | null; file_type: "TRAIN" | "TEST" | null; input_engine_count: number; screened_engine_count: number; excluded_insufficient_early_cycles: { engine_id: number; available_early_cycles: number }[]; domain_warning: string | null; };
  evaluation?: { mode: "prediction-only" | "evaluation"; available: boolean; message: string; dataset?: string; test_filename?: string; rul_filename?: string; aligned_engine_count?: number; target_definition?: string; rul_by_engine?: { engine_id: number; rul: number }[]; };
}
export interface DriftEvaluationResponse {
  description: string; target_sensor: string; target_cycle: number; input_window: { start_cycle: number; end_cycle: number };
  model_name: string; train_engine_count: number; validation_engine_count: number; test_engine_count: number;
  validation_mae: number; mae: number; recalculated_mae: number; baseline_model_name: string;
  baseline_validation_mae: number; baseline_mae: number; usable_engine_count: number; excluded_engine_count: number;
  exclusion_reason: string; engine_ids: number[]; predicted_values: number[]; actual_values: number[];
}
export interface ClearanceRequest { id: number; full_name: string; email: string; requested_role: "INSPECTOR" | "ENGINEER"; department: string; reason: string; status: "PENDING" | "APPROVED" | "DECLINED"; created_at: string; reviewed_at: string | null; reviewed_by_user_id: number | null; }
export interface AuditLog { id: number; event_type: string; actor_user_id: number | null; target_user_id: number | null; metadata: Record<string, unknown>; created_at: string; }
export interface ClearanceApprovalResponse {
  request: ClearanceRequest;
  activation_email_sent: boolean;
  email_delivery: { status: "sent" | "failed"; category: string; message: string; };
  activation_url?: string;
}

export class ApiError extends Error { constructor(message: string, public readonly status: number) { super(message); } }

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = configuredApiBaseUrl ? configuredApiBaseUrl.replace(/\/$/, "") : "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`${API_BASE_URL}${path}`, { credentials: "include", ...init }); }
  catch { throw new ApiError("Unable to connect to the BurnUI API. Check the API base URL and backend service.", 0); }
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(readApiError(payload, response.status), response.status);
  return payload as T;
}

export const getCurrentUser = (): Promise<{ user: AuthUser }> => request("/api/auth/me");
export const getPublicConfig = (): Promise<{ admin_contact_email: string | null; smtp_configured: boolean }> => request("/api/auth/public-config");
export const login = (email: string, password: string): Promise<{ user: AuthUser }> => request("/api/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
export const logout = (): Promise<{ message: string }> => request("/api/auth/logout", { method: "POST" });
export const activateAccount = (token: string, password: string): Promise<{ message: string }> => request("/api/auth/activate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token, password }) });
export const requestClearance = (payload: Omit<ClearanceRequest, "id" | "status" | "created_at" | "reviewed_at" | "reviewed_by_user_id">): Promise<{ request: ClearanceRequest }> => request("/api/auth/clearance-requests", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

export async function runScreening(file: File, parameters: ScreeningParameters, datasetVariant?: string, datasetSubset?: string, rulFile?: File | null): Promise<ScreeningResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("watchlist_threshold", String(parameters.watchlistThreshold));
  formData.append("high_risk_threshold", String(parameters.highRiskThreshold));
  formData.append("sensitivity", parameters.sensitivity);
  if (datasetVariant) formData.append("dataset_variant", datasetVariant);
  if (datasetSubset) formData.append("dataset_subset", datasetSubset);
  if (rulFile) formData.append("rul_file", rulFile);
  return request("/api/screening/run", { method: "POST", body: formData });
}
export async function runDriftEvaluation(): Promise<DriftEvaluationResponse> {
  const payload = await request<unknown>("/api/drift-evaluation/run");
  if (!isDriftEvaluationResponse(payload)) throw new ApiError("The drift evaluation API returned an unexpected response.", 502);
  return payload;
}
export const getAdminOverview = (): Promise<{ pending_clearance_requests: number; approved_clearance_requests: number; user_count: number; active_user_count: number }> => request("/api/admin/overview");
export const getClearanceRequests = (): Promise<{ requests: ClearanceRequest[] }> => request("/api/admin/clearance-requests");
export const getUsers = (): Promise<{ users: AuthUser[] }> => request("/api/admin/users");
export const getAuditLogs = (): Promise<{ audit_logs: AuditLog[] }> => request("/api/admin/audit-logs");
export const approveClearance = (id: number): Promise<ClearanceApprovalResponse> => request(`/api/admin/clearance-requests/${id}/approve`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }) });
export const declineClearance = (id: number): Promise<{ request: ClearanceRequest }> => request(`/api/admin/clearance-requests/${id}/decline`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }) });

function readApiError(payload: unknown, status: number): string {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (typeof detail === "object" && detail !== null && "message" in detail && typeof detail.message === "string") return detail.message;
  }
  return status === 0 ? "The API is unavailable." : `Request failed with status ${status}.`;
}

function isDriftEvaluationResponse(payload: unknown): payload is DriftEvaluationResponse {
  if (typeof payload !== "object" || payload === null) return false;
  const result = payload as Record<string, unknown>;
  const arrays = [result.engine_ids, result.predicted_values, result.actual_values];
  return typeof result.target_sensor === "string" && typeof result.target_cycle === "number" && typeof result.model_name === "string" && typeof result.mae === "number" && typeof result.baseline_mae === "number" && typeof result.input_window === "object" && result.input_window !== null && arrays.every((array) => Array.isArray(array) && array.every((value) => typeof value === "number" && Number.isFinite(value))) && (result.engine_ids as unknown[]).length === (result.predicted_values as unknown[]).length && (result.engine_ids as unknown[]).length === (result.actual_values as unknown[]).length;
}
