export type RiskStatus = "Normal" | "Watchlist" | "High Risk";

export interface TopFeature {
  feature: string;
  value: number;
  z_contribution: number;
}

export interface ScreenedComponent {
  engine_id: number;
  risk_score: number;
  status: RiskStatus;
  z_score: number;
  isolation_score: number;
  top_features: TopFeature[];
}

export interface ScreeningResponse {
  components: ScreenedComponent[];
  summary: {
    total_components_screened: number;
    normal_count: number;
    watchlist_count: number;
    high_risk_count: number;
  };
}

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = configuredApiBaseUrl ? configuredApiBaseUrl.replace(/\/$/, "") : "";

export async function runScreening(file: File): Promise<ScreeningResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/screening/run`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Unable to connect to the BurnUI API. Check the API base URL and backend service.");
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(readApiError(payload, response.status));
  }
  return payload as ScreeningResponse;
}

function readApiError(payload: unknown, status: number): string {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) {
    return `Screening request failed with status ${status}.`;
  }

  const detail = payload.detail;
  if (typeof detail === "string") return detail;
  if (typeof detail !== "object" || detail === null || !("message" in detail) || typeof detail.message !== "string") {
    return `Screening request failed with status ${status}.`;
  }

  const validation = "validation" in detail ? detail.validation : undefined;
  if (typeof validation === "object" && validation !== null && "errors" in validation && Array.isArray(validation.errors)) {
    const messages = validation.errors
      .map((issue) => (typeof issue === "object" && issue !== null && "message" in issue && typeof issue.message === "string" ? issue.message : null))
      .filter((message): message is string => message !== null);
    if (messages.length > 0) return `${detail.message} ${messages.join(" ")}`;
  }
  return detail.message;
}
