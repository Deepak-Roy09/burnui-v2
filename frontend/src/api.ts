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

export interface DriftEvaluationResponse {
  description: string;
  target_sensor: string;
  target_cycle: number;
  input_window: { start_cycle: number; end_cycle: number };
  model_name: string;
  train_engine_count: number;
  validation_engine_count: number;
  test_engine_count: number;
  validation_mae: number;
  mae: number;
  recalculated_mae: number;
  baseline_model_name: string;
  baseline_validation_mae: number;
  baseline_mae: number;
  usable_engine_count: number;
  excluded_engine_count: number;
  exclusion_reason: string;
  engine_ids: number[];
  predicted_values: number[];
  actual_values: number[];
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

export async function runDriftEvaluation(): Promise<DriftEvaluationResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/drift-evaluation/run`);
  } catch {
    throw new Error("Unable to connect to the BurnUI API. Check the API base URL and backend service.");
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(readApiError(payload, response.status, "Drift evaluation"));
  }
  if (!isDriftEvaluationResponse(payload)) {
    throw new Error("The drift evaluation API returned an unexpected response.");
  }
  return payload;
}

function readApiError(payload: unknown, status: number, requestName = "Screening"): string {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) {
    return `${requestName} request failed with status ${status}.`;
  }

  const detail = payload.detail;
  if (typeof detail === "string") return detail;
  if (typeof detail !== "object" || detail === null || !("message" in detail) || typeof detail.message !== "string") {
    return `${requestName} request failed with status ${status}.`;
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

function isDriftEvaluationResponse(payload: unknown): payload is DriftEvaluationResponse {
  if (typeof payload !== "object" || payload === null) return false;

  const response = payload as Record<string, unknown>;
  const numericFields = [
    "target_cycle", "train_engine_count", "validation_engine_count", "test_engine_count",
    "validation_mae", "mae", "recalculated_mae", "baseline_validation_mae", "baseline_mae",
    "usable_engine_count", "excluded_engine_count",
  ];
  const textFields = ["description", "target_sensor", "model_name", "baseline_model_name", "exclusion_reason"];
  const inputWindow = response.input_window;
  const engineIds = response.engine_ids;
  const predictedValues = response.predicted_values;
  const actualValues = response.actual_values;
  if (!Array.isArray(engineIds) || !Array.isArray(predictedValues) || !Array.isArray(actualValues)) {
    return false;
  }
  const predictionArrays = [engineIds, predictedValues, actualValues];
  const hasFinitePredictionArrays = predictionArrays.every((values) =>
    values.every((value) => typeof value === "number" && Number.isFinite(value)),
  );
  const matchingPredictionLengths =
    hasFinitePredictionArrays &&
    engineIds.length === predictedValues.length &&
    engineIds.length === actualValues.length &&
    engineIds.length === response.test_engine_count;

  return (
    numericFields.every((field) => typeof response[field] === "number" && Number.isFinite(response[field])) &&
    textFields.every((field) => typeof response[field] === "string") &&
    typeof inputWindow === "object" && inputWindow !== null &&
    Number.isFinite((inputWindow as Record<string, unknown>).start_cycle) &&
    Number.isFinite((inputWindow as Record<string, unknown>).end_cycle) &&
    matchingPredictionLengths
  );
}
