import { useRef, useState } from "react";
import { runScreening, type RiskStatus, type ScreenedComponent, type ScreeningResponse } from "./api";

const navigation = ["Dashboard", "Dataset", "Screening"] as const;
type View = (typeof navigation)[number];

const statusLabels: Record<RiskStatus, string> = {
  Normal: "Normal",
  Watchlist: "Watchlist",
  "High Risk": "High Risk",
};

export default function App() {
  const [view, setView] = useState<View>("Dashboard");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [results, setResults] = useState<ScreeningResponse | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<ScreenedComponent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleScreening() {
    if (!selectedFile) {
      setError("Choose a normalized C-MAPSS CSV before running screening.");
      return;
    }

    setError(null);
    setIsProcessing(true);
    try {
      const response = await runScreening(selectedFile);
      setResults(response);
      setSelectedComponent(response.components[0] ?? null);
      setView("Screening");
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "The screening request could not be completed.");
    } finally {
      setIsProcessing(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">B</span>
          <span>BurnUI</span>
        </div>
        <nav>
          {navigation.map((item) => (
            <button
              aria-current={view === item ? "page" : undefined}
              className={view === item ? "nav-item nav-item-active" : "nav-item"}
              key={item}
              type="button"
              onClick={() => setView(item)}
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>
      <main className="content">
        {view === "Dashboard" && <Dashboard results={results} onUpload={() => setView("Dataset")} />}
        {view === "Dataset" && (
          <DatasetUpload
            error={error}
            fileInputRef={fileInputRef}
            isProcessing={isProcessing}
            onChooseFile={(file) => {
              setSelectedFile(file);
              setError(null);
            }}
            onScreen={handleScreening}
            selectedFile={selectedFile}
          />
        )}
        {view === "Screening" && (
          <ScreeningResults
            onUpload={() => setView("Dataset")}
            results={results}
            selectedComponent={selectedComponent}
            onSelectComponent={setSelectedComponent}
          />
        )}
      </main>
    </div>
  );
}

function Dashboard({ results, onUpload }: { results: ScreeningResponse | null; onUpload: () => void }) {
  if (!results) {
    return (
      <section className="page-section empty-state">
        <p className="eyebrow">Screening dashboard</p>
        <h1>Component screening, ready for data.</h1>
        <p className="intro">Upload a normalized C-MAPSS CSV to run the existing real-only anomaly screen. Live component results will appear here after processing.</p>
        <button className="primary-button" type="button" onClick={onUpload}>Upload dataset</button>
      </section>
    );
  }

  return (
    <section className="page-section">
      <p className="eyebrow">Screening dashboard</p>
      <h1>Latest component risk profile.</h1>
      <p className="intro">Values below come from the most recent screening API response.</p>
      <SummaryCards summary={results.summary} />
    </section>
  );
}

function DatasetUpload({
  error,
  fileInputRef,
  isProcessing,
  onChooseFile,
  onScreen,
  selectedFile,
}: {
  error: string | null;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  isProcessing: boolean;
  onChooseFile: (file: File | null) => void;
  onScreen: () => void;
  selectedFile: File | null;
}) {
  return (
    <section className="page-section">
      <p className="eyebrow">Dataset intake</p>
      <h1>Screen a normalized sensor CSV.</h1>
      <p className="intro">The file is validated against the existing C-MAPSS-compatible contract before screening. It is not stored by the API.</p>
      <div className="upload-panel">
        <input
          ref={fileInputRef}
          className="visually-hidden"
          accept=".csv,text/csv"
          id="dataset-file"
          type="file"
          onChange={(event) => onChooseFile(event.target.files?.[0] ?? null)}
        />
        <label className="file-picker" htmlFor="dataset-file">
          <span className="file-picker-label">{selectedFile ? "Selected CSV" : "Choose CSV file"}</span>
          <span className="file-name">{selectedFile?.name ?? "No file selected"}</span>
        </label>
        <button className="primary-button" disabled={isProcessing} type="button" onClick={onScreen}>
          {isProcessing ? "Validating and screening…" : "Run screening"}
        </button>
      </div>
      {error && <p className="error-message" role="alert">{error}</p>}
      <p className="contract-note">Required columns: engine_id, cycle, setting_1–3, and sequential sensor_1…sensor_N. Every screened engine needs 30 early cycles.</p>
    </section>
  );
}

function ScreeningResults({
  onSelectComponent,
  onUpload,
  results,
  selectedComponent,
}: {
  onSelectComponent: (component: ScreenedComponent) => void;
  onUpload: () => void;
  results: ScreeningResponse | null;
  selectedComponent: ScreenedComponent | null;
}) {
  if (!results) {
    return (
      <section className="page-section empty-state">
        <p className="eyebrow">Screening results</p>
        <h1>No screening results yet.</h1>
        <p className="intro">Upload and screen a CSV dataset to review component risk scores and signals.</p>
        <button className="primary-button" type="button" onClick={onUpload}>Go to dataset upload</button>
      </section>
    );
  }

  return (
    <section className="page-section results-section">
      <p className="eyebrow">Screening results</p>
      <h1>Component risk screening.</h1>
      <SummaryCards summary={results.summary} />
      <div className="screening-grid">
        <div className="table-panel">
          <div className="panel-heading">
            <h2>Screened components</h2>
            <span>{results.summary.total_components_screened} total</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr><th>Component / Engine ID</th><th>Risk Score</th><th>Status</th><th>Main Signal</th></tr>
              </thead>
              <tbody>
                {results.components.map((component) => (
                  <tr className={selectedComponent?.engine_id === component.engine_id ? "selected-row" : undefined} key={component.engine_id} onClick={() => onSelectComponent(component)}>
                    <td>Engine {component.engine_id}</td>
                    <td>{formatScore(component.risk_score)}</td>
                    <td><StatusBadge status={component.status} /></td>
                    <td>{component.top_features[0]?.feature ?? "No signal returned"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <ComponentDetails component={selectedComponent} />
      </div>
    </section>
  );
}

function SummaryCards({ summary }: { summary: ScreeningResponse["summary"] }) {
  const cards = [
    ["Components screened", summary.total_components_screened, "neutral"],
    ["Normal", summary.normal_count, "normal"],
    ["Watchlist", summary.watchlist_count, "watchlist"],
    ["High Risk", summary.high_risk_count, "high-risk"],
  ] as const;
  return <div className="summary-grid">{cards.map(([label, value, tone]) => <div className={`summary-card ${tone}`} key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>;
}

function ComponentDetails({ component }: { component: ScreenedComponent | null }) {
  if (!component) return null;
  return (
    <aside className="details-panel" aria-label="Selected component details">
      <div className="panel-heading"><h2>Component details</h2><StatusBadge status={component.status} /></div>
      <dl className="signal-list">
        <div><dt>Engine ID</dt><dd>{component.engine_id}</dd></div>
        <div><dt>Risk score</dt><dd>{formatScore(component.risk_score)}</dd></div>
        <div><dt>Z-score signal</dt><dd>{formatScore(component.z_score)}</dd></div>
        <div><dt>Isolation Forest signal</dt><dd>{formatScore(component.isolation_score)}</dd></div>
      </dl>
      <section className="status-explanation">
        <h3>Why this status?</h3>
        <p>{buildStatusExplanation(component)}</p>
        <p className="screening-disclaimer">Screening indicates unusual early-cycle behavior; it is not a confirmed component failure.</p>
      </section>
      <h3>Top signals</h3>
      <ul className="feature-list">
        {component.top_features.map((feature) => <li key={feature.feature}><strong>{feature.feature}</strong><span>Value {formatScore(feature.value)} · Z contribution {formatScore(feature.z_contribution)}</span></li>)}
      </ul>
    </aside>
  );
}

function StatusBadge({ status }: { status: RiskStatus }) {
  return <span className={`status-badge status-${status.toLowerCase().replace(" ", "-")}`}>{statusLabels[status]}</span>;
}

function formatScore(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function buildStatusExplanation(component: ScreenedComponent): string {
  const strongestFeature = component.top_features[0];
  const featureDetail = strongestFeature
    ? `The strongest returned feature is ${strongestFeature.feature} (value ${formatScore(strongestFeature.value)}; contribution ${formatScore(strongestFeature.z_contribution)}).`
    : "No individual top feature was returned for this component.";
  const signalDetail = `The early-pattern difference signal is ${formatScore(component.z_score)}, and the anomaly-detection signal is ${formatScore(component.isolation_score)}; both are included in the current screening score.`;

  if (component.status === "High Risk") {
    return `The current screening thresholds classify this component as High Risk. ${featureDetail} ${signalDetail} Together, these signals indicate an unusual early-cycle pattern.`;
  }
  if (component.status === "Watchlist") {
    return `The current screening thresholds place this component on the Watchlist because moderate abnormal behavior was detected. ${featureDetail} ${signalDetail} Monitor this component in follow-up screening rather than treating this result as a failure.`;
  }
  return `No strong anomaly was detected by the current screening model, so this component remains Normal under the current thresholds. ${featureDetail} ${signalDetail} The resulting score is below the Watchlist threshold.`;
}
