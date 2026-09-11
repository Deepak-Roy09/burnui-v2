import { useEffect, useRef, useState } from "react";
import {
  activateAccount, ApiError, approveClearance, declineClearance, getAdminOverview,
  getAuditLogs, getClearanceRequests, getCurrentUser, getPublicConfig, getUsers, login, logout,
  requestClearance, runDriftEvaluation, runScreening,
  type AuditLog, type AuthUser, type ClearanceRequest,
  type DriftEvaluationResponse, type RiskStatus, type ScreenedComponent, type ScreeningParameters,
  type ScreeningResponse,
} from "./api";

const DEFAULT_PARAMETERS: ScreeningParameters = {
  sensitivity: "Standard",
  watchlistThreshold: 0.807143,
  highRiskThreshold: 0.885714,
};
const SENSITIVITY_PRESETS: Record<ScreeningParameters["sensitivity"], Pick<ScreeningParameters, "watchlistThreshold" | "highRiskThreshold">> = {
  Standard: { watchlistThreshold: 0.807143, highRiskThreshold: 0.885714 },
  Sensitive: { watchlistThreshold: 0.75, highRiskThreshold: 0.85 },
  Conservative: { watchlistThreshold: 0.85, highRiskThreshold: 0.93 },
};
const statusLabels: Record<RiskStatus, string> = { Normal: "Normal", Watchlist: "Watchlist", "High Risk": "High Risk" };
type View = "DATASET" | "ANALYSIS" | "EVALUATION" | "SETTINGS" | "ADMIN_OVERVIEW" | "ADMIN_REQUESTS" | "ADMIN_USERS" | "ADMIN_AUDIT";

export default function App() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isAuthenticating, setIsAuthenticating] = useState(true);
  const [authError, setAuthError] = useState<string | null>(null);
  const [activationHandled, setActivationHandled] = useState(false);
  const [view, setView] = useState<View>("DATASET");
  const [results, setResults] = useState<ScreeningResponse | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<ScreenedComponent | null>(null);

  useEffect(() => {
    let active = true;
    getCurrentUser()
      .then(({ user: existingUser }) => { if (active) { setUser(existingUser); setView(existingUser.role === "ADMIN" ? "ADMIN_OVERVIEW" : "DATASET"); } })
      .catch((error: unknown) => { if (active && (!(error instanceof ApiError) || error.status !== 401)) setAuthError(error instanceof Error ? error.message : "Authentication is unavailable."); })
      .finally(() => { if (active) setIsAuthenticating(false); });
    return () => { active = false; };
  }, []);

  const activationToken = activationHandled ? null : new URLSearchParams(window.location.search).get("activate");
  if (isAuthenticating) return <main className="auth-shell"><p className="auth-loading">Checking secure session…</p></main>;
  if (activationToken) return <ActivationPage token={activationToken} onActivated={() => { window.history.replaceState({}, "", window.location.pathname); setActivationHandled(true); }} />;
  if (!user) return <LoginPage startupError={authError} onLogin={(loggedIn) => { setUser(loggedIn); setView(loggedIn.role === "ADMIN" ? "ADMIN_OVERVIEW" : "DATASET"); }} />;

  async function handleLogout() {
    try { await logout(); } catch { /* Expired sessions are already invalidated server-side. */ }
    setUser(null); setResults(null); setSelectedComponent(null); setView("DATASET");
  }
  const openComponent = (component: ScreenedComponent) => { setSelectedComponent(component); setView("ANALYSIS"); };

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <div className="brand"><span className="brand-mark" aria-hidden="true">B</span><span>BurnUI</span></div>
        <nav>
          <NavItem label="Screening" active={view === "DATASET"} onClick={() => setView("DATASET")} />
          <NavItem label="Dashboard" active={view === "ANALYSIS"} onClick={() => setView("ANALYSIS")} />
          <NavItem label="Evaluation / Drift" active={view === "EVALUATION"} onClick={() => setView("EVALUATION")} />
          <NavItem label="Settings" active={view === "SETTINGS"} onClick={() => setView("SETTINGS")} />
          {user.role === "ADMIN" && <>
            <p className="nav-section">Administration</p>
            <NavItem label="Admin Overview" active={view === "ADMIN_OVERVIEW"} onClick={() => setView("ADMIN_OVERVIEW")} />
            <NavItem label="Clearance Requests" active={view === "ADMIN_REQUESTS"} onClick={() => setView("ADMIN_REQUESTS")} />
            <NavItem label="Users" active={view === "ADMIN_USERS"} onClick={() => setView("ADMIN_USERS")} />
            <NavItem label="Audit Logs" active={view === "ADMIN_AUDIT"} onClick={() => setView("ADMIN_AUDIT")} />
          </>}
        </nav>
        <div className="session-card"><strong>{user.full_name}</strong><span>{user.role}</span><button type="button" className="logout-button" onClick={handleLogout}>Logout</button></div>
      </aside>
      <main className="content">
        {view === "DATASET" && <DatasetScreen onFileSelected={() => { setResults(null); setSelectedComponent(null); }} onResults={(response) => { setResults(response); setSelectedComponent(response.components[0] ?? null); setView("ANALYSIS"); }} />}
        {view === "ANALYSIS" && <ComponentAnalysis results={results} selected={selectedComponent} onSelect={setSelectedComponent} />}
        {view === "EVALUATION" && <ModelEvaluation results={results} onOpenComponent={openComponent} />}
        {view === "SETTINGS" && <SettingsPage />}
        {user.role === "ADMIN" && view.startsWith("ADMIN_") && <AdminPage view={view} />}
      </main>
    </div>
  );
}

function NavItem({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return <button type="button" className={active ? "nav-item nav-item-active" : "nav-item"} aria-current={active ? "page" : undefined} onClick={onClick}>{label}</button>;
}

function LoginPage({ onLogin, startupError }: { onLogin: (user: AuthUser) => void; startupError: string | null }) {
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [error, setError] = useState<string | null>(startupError); const [busy, setBusy] = useState(false); const [showClearance, setShowClearance] = useState(false); const [adminEmail, setAdminEmail] = useState<string | null>(null);
  useEffect(() => { getPublicConfig().then((config) => setAdminEmail(config.admin_contact_email)).catch(() => setAdminEmail(null)); }, []);
  async function submit(event: React.FormEvent) { event.preventDefault(); setBusy(true); setError(null); try { const response = await login(email, password); onLogin(response.user); } catch (caught) { setError(caught instanceof Error ? caught.message : "Unable to sign in."); } finally { setBusy(false); } }
  return <main className="auth-shell"><section className="auth-panel"><div className="brand auth-brand"><span className="brand-mark">B</span><span>BurnUI</span></div><p className="eyebrow">Secure screening workspace</p><h1>Sign in to continue.</h1><p className="intro">Inspectors and engineers are taken directly to Dataset / Screening after login.</p><form className="auth-form" onSubmit={submit}><label>Email<input autoComplete="email" required type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label><label>Password<input autoComplete="current-password" required type="password" value={password} onChange={(event) => setPassword(event.target.value)} /></label><button className="primary-button" disabled={busy} type="submit">{busy ? "Signing in…" : "Sign in"}</button></form>{error && <p className="error-message" role="alert">{error}</p>}<section className="forgot-section"><h2>Forgot your password?</h2><p>For security, password recovery is handled by the administrator.</p>{adminEmail ? <a href={`mailto:${adminEmail}`}>{adminEmail}</a> : <span>Administrator contact is not configured.</span>}</section><button type="button" className="link-button" onClick={() => setShowClearance((shown) => !shown)}>{showClearance ? "Hide request access" : "Request Access / Clearance"}</button>{showClearance && <ClearanceForm />}</section></main>;
}

function ClearanceForm() {
  const [message, setMessage] = useState<string | null>(null); const [error, setError] = useState<string | null>(null); const [busy, setBusy] = useState(false);
  async function submit(event: React.FormEvent<HTMLFormElement>) { event.preventDefault(); const form = new FormData(event.currentTarget); setBusy(true); setError(null); setMessage(null); try { const response = await requestClearance({ full_name: String(form.get("full_name")), email: String(form.get("email")), requested_role: String(form.get("requested_role")) as "INSPECTOR" | "ENGINEER", department: String(form.get("department")), reason: String(form.get("reason")) }); setMessage(`Clearance request #${response.request.id} is PENDING.`); event.currentTarget.reset(); } catch (caught) { setError(caught instanceof Error ? caught.message : "Unable to submit clearance request."); } finally { setBusy(false); } }
  return <form className="clearance-form" onSubmit={submit}><h2>Request Clearance</h2><label>Full name<input name="full_name" required minLength={2} /></label><label>Email<input name="email" required type="email" /></label><label>Requested role<select name="requested_role" defaultValue="INSPECTOR"><option value="INSPECTOR">Inspector</option><option value="ENGINEER">Engineer</option></select></label><label>Department<input name="department" required minLength={2} /></label><label>Reason<textarea name="reason" required minLength={5} rows={3} /></label><button className="primary-button" disabled={busy} type="submit">{busy ? "Submitting…" : "Submit clearance request"}</button>{message && <p className="success-message">{message}</p>}{error && <p className="error-message" role="alert">{error}</p>}</form>;
}

function ActivationPage({ token, onActivated }: { token: string; onActivated: () => void }) {
  const [password, setPassword] = useState(""); const [confirm, setConfirm] = useState(""); const [message, setMessage] = useState<string | null>(null); const [error, setError] = useState<string | null>(null); const [busy, setBusy] = useState(false);
  async function submit(event: React.FormEvent) { event.preventDefault(); if (password !== confirm) { setError("Passwords do not match."); return; } setBusy(true); setError(null); try { const response = await activateAccount(token, password); setMessage(response.message); onActivated(); } catch (caught) { setError(caught instanceof Error ? caught.message : "Account activation failed."); } finally { setBusy(false); } }
  return <main className="auth-shell"><section className="auth-panel"><p className="eyebrow">Account activation</p><h1>Create your password.</h1><p className="intro">This secure activation link is single-use and time-limited.</p><form className="auth-form" onSubmit={submit}><label>New password<input autoComplete="new-password" minLength={12} required type="password" value={password} onChange={(event) => setPassword(event.target.value)} /></label><label>Confirm password<input autoComplete="new-password" minLength={12} required type="password" value={confirm} onChange={(event) => setConfirm(event.target.value)} /></label><button className="primary-button" disabled={busy} type="submit">{busy ? "Activating…" : "Activate account"}</button></form>{message && <p className="success-message">{message}</p>}{error && <p className="error-message" role="alert">{error}</p>}</section></main>;
}

function DatasetScreen({ onResults, onFileSelected }: { onResults: (results: ScreeningResponse) => void; onFileSelected: () => void }) {
  const [file, setFile] = useState<File | null>(null); const [parameters, setParameters] = useState(DEFAULT_PARAMETERS); const [error, setError] = useState<string | null>(null); const [busy, setBusy] = useState(false); const [success, setSuccess] = useState<string | null>(null); const fileInputRef = useRef<HTMLInputElement>(null);
  function chooseSensitivity(sensitivity: ScreeningParameters["sensitivity"]) { setParameters({ sensitivity, ...SENSITIVITY_PRESETS[sensitivity] }); }
  function chooseFile(nextFile: File | null) { setFile(nextFile); setError(null); setSuccess(null); onFileSelected(); }
  async function screen() { if (!file) { setError("Choose a normalized CSV or NASA C-MAPSS train/test TXT file before screening."); return; } if (!Number.isFinite(parameters.watchlistThreshold) || !Number.isFinite(parameters.highRiskThreshold) || parameters.watchlistThreshold < 0 || parameters.highRiskThreshold > 1 || parameters.highRiskThreshold <= parameters.watchlistThreshold) { setError("Use finite thresholds from 0 to 1, with High Risk greater than Watchlist."); return; } setBusy(true); setError(null); setSuccess(null); try { const response = await runScreening(file, parameters); onResults(response); setParameters(DEFAULT_PARAMETERS); setSuccess("Screening completed. Parameters were reset to validated production defaults."); } catch (caught) { setError(caught instanceof Error ? caught.message : "Screening could not be completed."); } finally { setBusy(false); } }
  return <section className="page-section"><p className="eyebrow">Production screening</p><h1>Dataset Upload / Screening</h1><p className="intro">The saved FD001 production model uses existing cycles 1–30 and 168 sensor-derived features. Uploads do not retrain the model.</p><div className="dataset-layout"><div className="upload-panel"><input ref={fileInputRef} className="visually-hidden" accept=".csv,.txt,text/csv,text/plain" id="dataset-file" type="file" onChange={(event) => chooseFile(event.target.files?.[0] ?? null)} /><label className="file-picker" htmlFor="dataset-file"><span className="file-picker-label">{file ? "Selected file" : "Choose CSV or C-MAPSS TXT"}</span><span className="file-name">{file?.name ?? "No file selected"}</span></label><button className="primary-button" disabled={busy} type="button" onClick={screen}>{busy ? "Validating and screening…" : "Run screening"}</button></div><ScreeningParameterPanel parameters={parameters} onChange={setParameters} onSensitivity={chooseSensitivity} onReset={() => setParameters(DEFAULT_PARAMETERS)} /></div>{success && <p className="success-message">{success}</p>}{error && <p className="error-message" role="alert">{error}</p>}<p className="cmaps-disclaimer">C-MAPSS is used as a public degradation/prognostics proxy. It is not actual ISRO component burn-in data. The prototype can later be adapted to approved real burn-in measurements only after retraining and validation on representative component data.</p></section>;
}

function ScreeningParameterPanel({ parameters, onChange, onSensitivity, onReset }: { parameters: ScreeningParameters; onChange: (parameters: ScreeningParameters) => void; onSensitivity: (sensitivity: ScreeningParameters["sensitivity"]) => void; onReset: () => void }) {
  return <aside className="parameter-panel"><p className="section-label">Screening Parameters</p><strong>Validated production defaults</strong><label>Sensitivity<select value={parameters.sensitivity} onChange={(event) => onSensitivity(event.target.value as ScreeningParameters["sensitivity"])}><option>Standard</option><option>Sensitive</option><option>Conservative</option></select></label><label>Watchlist threshold<input type="number" min="0" max="1" step="0.000001" value={parameters.watchlistThreshold} onChange={(event) => onChange({ ...parameters, watchlistThreshold: Number(event.target.value) })} /></label><label>High Risk threshold<input type="number" min="0" max="1" step="0.000001" value={parameters.highRiskThreshold} onChange={(event) => onChange({ ...parameters, highRiskThreshold: Number(event.target.value) })} /></label><button className="link-button" type="button" onClick={onReset}>Reset to validated defaults</button><p>Changing thresholds changes screening sensitivity; evaluation metrics are based on the validated evaluation configuration.</p></aside>;
}

function ComponentAnalysis({ results, selected, onSelect }: { results: ScreeningResponse | null; selected: ScreenedComponent | null; onSelect: (component: ScreenedComponent) => void }) {
  if (!results) return <section className="page-section empty-state"><p className="eyebrow">Screening dashboard</p><h1>No screening results yet.</h1><p className="intro">Upload a normalized CSV or C-MAPSS file to review component decisions and their returned evidence.</p></section>;
  const predictionOnly = !results.evaluation || results.evaluation.mode === "prediction-only";
  return <section className="page-section results-section"><p className="eyebrow">Production screening</p><h1>Screening dashboard</h1><div className="summary-grid"><SummaryCard label="Screened" value={results.summary.total_components_screened} /><SummaryCard label="Normal" value={results.summary.normal_count} tone="normal" /><SummaryCard label="Watchlist" value={results.summary.watchlist_count} tone="watchlist" /><SummaryCard label="High Risk" value={results.summary.high_risk_count} tone="high-risk" /></div>{results.source && <section className="source-summary"><strong>Dataset</strong><span>{results.source.dataset ?? "Normalized CSV"}</span><strong>File</strong><span>{results.source.filename}</span><strong>Scope</strong><span>{results.source.file_type ?? "Uploaded dataset"} · {results.source.screened_engine_count} components screened</span>{results.source.domain_warning && <p>{results.source.domain_warning}</p>}</section>}{predictionOnly ? <p className="prediction-only" role="status">Prediction only — no ground-truth outcome supplied.</p> : <details className="evaluation-details"><summary>Evaluation details</summary><p>{results.evaluation?.message ?? results.evaluation?.target_definition ?? "Ground-truth evaluation is available for this upload."}</p></details>}<div className="screening-grid"><div className="table-panel"><div className="panel-heading"><h2>Components</h2><span>Select a row for evidence</span></div><div className="table-scroll"><table><thead><tr><th>Component</th><th>Risk score</th><th>Status</th><th>Key reason</th></tr></thead><tbody>{results.components.map((component) => <tr className={selected?.engine_id === component.engine_id ? "selected-row" : undefined} key={component.engine_id} onClick={() => onSelect(component)}><td>Engine {component.engine_id}</td><td>{formatNumber(component.risk_score, 3)}</td><td><StatusBadge status={component.status} /></td><td>{component.top_features[0]?.feature ?? "No signal returned"}</td></tr>)}</tbody></table></div></div><ComponentDetails component={selected} /></div></section>;
}

function SummaryCard({ label, value, tone }: { label: string; value: number; tone?: "normal" | "watchlist" | "high-risk" }) { return <article className={`summary-card ${tone ?? ""}`}><span>{label}</span><strong>{value}</strong></article>; }

function ComponentDetails({ component }: { component: ScreenedComponent | null }) {
  if (!component) return <aside className="details-panel"><p className="muted">Select a component to inspect its returned screening evidence.</p></aside>;
  return <aside className="details-panel" aria-label="Selected component details"><div className="panel-heading"><h2>Component details</h2><StatusBadge status={component.status} /></div><dl className="signal-list"><div><dt>Component ID</dt><dd>{component.engine_id}</dd></div><div><dt>Risk score</dt><dd>{formatNumber(component.risk_score, 3)}</dd></div></dl><section className="status-explanation"><h3>Why this status?</h3><p>{statusExplanation(component)}</p><p className="screening-disclaimer">Screening indicates unusual early-cycle behavior; it is not a confirmed component failure.</p></section><details className="technical-details"><summary>View returned signal details</summary><dl className="signal-list"><div><dt>Z-score evidence</dt><dd>{formatNumber(component.z_score, 3)}</dd></div><div><dt>Isolation Forest evidence</dt><dd>{formatNumber(component.isolation_score, 3)}</dd></div></dl><h3>Strongest returned signals</h3><ul className="feature-list">{component.top_features.map((feature) => <li key={feature.feature}><strong>{feature.feature}</strong><span>Value {formatNumber(feature.value, 3)} · Z contribution {formatNumber(feature.z_contribution, 3)}</span></li>)}</ul></details></aside>;
}

function ModelEvaluation({ results, onOpenComponent }: { results: ScreeningResponse | null; onOpenComponent: (component: ScreenedComponent) => void }) {
  const [drift, setDrift] = useState<DriftEvaluationResponse | null>(null); const [driftError, setDriftError] = useState<string | null>(null); const [loading, setLoading] = useState(true);
  useEffect(() => { let active = true; runDriftEvaluation().then((result) => { if (active) setDrift(result); }).catch((error: unknown) => { if (active) setDriftError(error instanceof Error ? error.message : "Drift evaluation is unavailable."); }).finally(() => { if (active) setLoading(false); }); return () => { active = false; }; }, []);
  return <section className="page-section"><p className="eyebrow">Model evaluation</p><h1>Evidence, separate from live screening.</h1><p className="intro">C-MAPSS is a public degradation/prognostics proxy, not actual ISRO component burn-in data. These visuals do not change production risk decisions.</p>{loading && <div className="loading-state" role="status"><span className="loading-spinner" />Loading evaluation data…</div>}{!loading && <div className="visualization-grid"><section className="visualization-section"><div className="visualization-heading"><div><h2>Component Behavioral Map</h2><p>PCA visualization — not a clustering algorithm.</p></div></div><OutlierMap results={results} onSelect={onOpenComponent} /><p className="chart-disclaimer">One point represents one screened component. Color reflects the existing production screening status only.</p></section><section className="visualization-section"><div className="visualization-heading"><div><h2>Drift: Predicted vs Actual</h2><p>Separate drift-prediction evaluation.</p></div></div>{driftError && <p className="error-message" role="alert">Drift evaluation unavailable. {driftError}</p>}<DriftScatter evaluation={drift} /></section></div>}</section>;
}

function SettingsPage() { return <section className="page-section settings-page"><div className="settings-empty-state"><p className="eyebrow">Workspace settings</p><h1>Settings</h1><h2>Coming soon</h2><p>Configuration and personalization options will be available here in a future release.</p></div></section>; }

function OutlierMap({ results, onSelect }: { results: ScreeningResponse | null; onSelect: (component: ScreenedComponent) => void }) {
  const points = results?.components.filter((component) => Number.isFinite(component.pca_1) && Number.isFinite(component.pca_2)) ?? [];
  if (!points.length) return <p className="muted">Run a screening upload to generate the component-level visualization from its existing 168 early-cycle features.</p>;
  const x = points.map((point) => point.pca_1 ?? 0); const y = points.map((point) => point.pca_2 ?? 0); const [minX, maxX] = paddedExtent(x); const [minY, maxY] = paddedExtent(y); const px = (value: number) => 58 + ((value - minX) / (maxX - minX)) * 522; const py = (value: number) => 252 - ((value - minY) / (maxY - minY)) * 202;
  return <div className="chart-wrap"><svg className="evidence-chart" viewBox="0 0 640 300" role="img" aria-label="PCA component-level outlier map"><title>Component-Level Outlier Map</title><line className="chart-axis" x1="58" x2="580" y1="252" y2="252" /><line className="chart-axis" x1="58" x2="58" y1="34" y2="252" /><text x="320" y="288" textAnchor="middle">PCA 1</text><text x="16" y="146" textAnchor="middle" transform="rotate(-90 16 146)">PCA 2</text>{points.map((point) => <circle key={point.engine_id} className={`outlier-point point-${point.status.toLowerCase().replace(" ", "-")}`} cx={px(point.pca_1 ?? 0)} cy={py(point.pca_2 ?? 0)} r="6" tabIndex={0} role="button" aria-label={`Engine ${point.engine_id}, ${point.status}, risk ${formatNumber(point.risk_score, 3)}, PCA 1 ${formatNumber(point.pca_1 ?? 0, 3)}, PCA 2 ${formatNumber(point.pca_2 ?? 0, 3)}`} onClick={() => onSelect(point)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onSelect(point); }}><title>Engine {point.engine_id} · {point.status} · Risk {formatNumber(point.risk_score, 3)} · PCA 1 {formatNumber(point.pca_1 ?? 0, 3)} · PCA 2 {formatNumber(point.pca_2 ?? 0, 3)}</title></circle>)}</svg><div className="chart-legend"><span><i className="legend-dot normal-dot" />Normal</span><span><i className="legend-dot watchlist-dot" />Watchlist</span><span><i className="legend-dot high-risk-dot" />High Risk</span></div></div>;
}

function DriftScatter({ evaluation }: { evaluation: DriftEvaluationResponse | null }) {
  if (!evaluation) return <p className="muted">Drift evaluation is unavailable.</p>;
  const values = [...evaluation.actual_values, ...evaluation.predicted_values]; const [minimum, maximum] = paddedExtent(values); const pointX = (value: number) => 58 + ((value - minimum) / (maximum - minimum)) * 522; const pointY = (value: number) => 252 - ((value - minimum) / (maximum - minimum)) * 202;
  return <div className="chart-wrap"><svg className="evidence-chart" viewBox="0 0 640 300" role="img" aria-label="Predicted versus actual sensor 12 at cycle 50"><title>Drift predicted versus actual</title><line className="chart-axis" x1="58" x2="580" y1="252" y2="252" /><line className="chart-axis" x1="58" x2="58" y1="34" y2="252" /><line className="ideal-line" x1={pointX(minimum)} y1={pointY(minimum)} x2={pointX(maximum)} y2={pointY(maximum)} />{evaluation.actual_values.map((actual, index) => <circle className="drift-point" key={evaluation.engine_ids[index]} cx={pointX(actual)} cy={pointY(evaluation.predicted_values[index])} r="3"><title>Engine {evaluation.engine_ids[index]} · actual {formatNumber(actual, 4)} · predicted {formatNumber(evaluation.predicted_values[index], 4)}</title></circle>)}<text x="320" y="288" textAnchor="middle">Actual sensor_12 at cycle 50</text><text x="16" y="146" textAnchor="middle" transform="rotate(-90 16 146)">Predicted sensor_12 at cycle 50</text></svg><div className="chart-legend"><span><i className="legend-line" />Ideal y=x</span><span><i className="legend-dot drift-dot" />Predicted observation</span></div><div className="drift-summary"><span>Test MAE {formatNumber(evaluation.mae, 5)}</span><span>Baseline MAE {formatNumber(evaluation.baseline_mae, 5)}</span></div></div>;
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="metric-card"><span>{label}</span><strong>{value}</strong></div>; }
function StatusBadge({ status }: { status: RiskStatus }) { return <span className={`status-badge status-${status.toLowerCase().replace(" ", "-")}`}>{statusLabels[status]}</span>; }

function AdminPage({ view }: { view: View }) {
  const [overview, setOverview] = useState<{ pending_clearance_requests: number; approved_clearance_requests: number; user_count: number; active_user_count: number } | null>(null); const [requests, setRequests] = useState<ClearanceRequest[]>([]); const [users, setUsers] = useState<AuthUser[]>([]); const [logs, setLogs] = useState<AuditLog[]>([]); const [error, setError] = useState<string | null>(null); const [busyId, setBusyId] = useState<number | null>(null); const [approvalNotice, setApprovalNotice] = useState<string | null>(null); const [activationUrl, setActivationUrl] = useState<string | null>(null);
  const load = async () => { setError(null); try { if (view === "ADMIN_OVERVIEW") setOverview(await getAdminOverview()); if (view === "ADMIN_REQUESTS") setRequests((await getClearanceRequests()).requests); if (view === "ADMIN_USERS") setUsers((await getUsers()).users); if (view === "ADMIN_AUDIT") setLogs((await getAuditLogs()).audit_logs); } catch (caught) { setError(caught instanceof Error ? caught.message : "Administration data is unavailable."); } };
  useEffect(() => { void load(); }, [view]);
  async function review(id: number, action: "approve" | "decline") { if (!window.confirm(`${action === "approve" ? "Approve" : "Decline"} this clearance request?`)) return; setBusyId(id); setApprovalNotice(null); setActivationUrl(null); try { if (action === "approve") { const result = await approveClearance(id); setApprovalNotice(result.email_delivery.message); setActivationUrl(result.activation_url ?? null); } else { await declineClearance(id); } await load(); } catch (caught) { setError(caught instanceof Error ? caught.message : "Unable to review request."); } finally { setBusyId(null); } }
  const titles: Record<string, string> = { ADMIN_OVERVIEW: "Admin Overview", ADMIN_REQUESTS: "Clearance Requests", ADMIN_USERS: "Users", ADMIN_AUDIT: "Audit Logs" };
  return <section className="page-section"><p className="eyebrow">Protected administration</p><h1>{titles[view]}</h1>{error && <p className="error-message" role="alert">{error}</p>}{approvalNotice && <div className="approval-notice" role="status"><p>{approvalNotice}</p>{activationUrl && <a href={activationUrl} target="_blank" rel="noreferrer">Open secure activation link</a>}</div>}{view === "ADMIN_OVERVIEW" && overview && <div className="metric-grid"><Metric label="Pending requests" value={String(overview.pending_clearance_requests)} /><Metric label="Approved requests" value={String(overview.approved_clearance_requests)} /><Metric label="Users" value={String(overview.user_count)} /><Metric label="Active users" value={String(overview.active_user_count)} /></div>}{view === "ADMIN_REQUESTS" && <div className="table-panel table-scroll"><table><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Department</th><th>Reason</th><th>Date</th><th>Status</th><th>Actions</th></tr></thead><tbody>{requests.map((request) => <tr key={request.id}><td>{request.full_name}</td><td>{request.email}</td><td>{request.requested_role}</td><td>{request.department}</td><td className="wrap-cell">{request.reason}</td><td>{formatDate(request.created_at)}</td><td>{request.status}</td><td>{request.status === "PENDING" && <span className="action-row"><button disabled={busyId === request.id} onClick={() => review(request.id, "approve")} type="button">Approve</button><button disabled={busyId === request.id} onClick={() => review(request.id, "decline")} type="button">Decline</button></span>}</td></tr>)}</tbody></table></div>}{view === "ADMIN_USERS" && <div className="table-panel table-scroll"><table><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Department</th><th>Active</th><th>Created</th></tr></thead><tbody>{users.map((account) => <tr key={account.id}><td>{account.full_name}</td><td>{account.email}</td><td>{account.role}</td><td>{account.department ?? "—"}</td><td>{account.is_active ? "Yes" : "Activation pending"}</td><td>{formatDate(account.created_at)}</td></tr>)}</tbody></table></div>}{view === "ADMIN_AUDIT" && <div className="table-panel table-scroll"><table><thead><tr><th>Time</th><th>Event</th><th>Actor</th><th>Target</th><th>Metadata</th></tr></thead><tbody>{logs.map((log) => <tr key={log.id}><td>{formatDate(log.created_at)}</td><td>{log.event_type}</td><td>{log.actor_user_id ?? "—"}</td><td>{log.target_user_id ?? "—"}</td><td>{JSON.stringify(log.metadata)}</td></tr>)}</tbody></table></div>}</section>;
}

function formatNumber(value: number, digits: number) { return Number.isFinite(value) ? value.toFixed(digits) : "—"; }
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString(); }
function paddedExtent(values: number[]): [number, number] { const low = Math.min(...values); const high = Math.max(...values); if (!Number.isFinite(low) || !Number.isFinite(high) || low === high) return [low - 1, high + 1]; const pad = (high - low) * 0.08; return [low - pad, high + pad]; }
function statusExplanation(component: ScreenedComponent) { const top = component.top_features[0]; const featureText = top ? `The strongest returned signal is ${top.feature} with a calculated Z-score contribution of ${formatNumber(top.z_contribution, 3)}.` : "No individual feature contribution was returned."; if (component.status === "Normal") return `This component has no strong anomaly under the selected decision thresholds. ${featureText}`; return `This component shows unusual early-cycle behaviour relative to the learned normal pattern. ${featureText} The Z-score evidence is ${formatNumber(component.z_score, 3)} and the Isolation Forest evidence is ${formatNumber(component.isolation_score, 3)}.`; }
