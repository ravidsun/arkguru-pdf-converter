import type { Defaults } from "../types";

export interface Phase1State {
  input_dir: string;
  out_dir: string;
  backend: "pymupdf4llm" | "docling" | "pymupdf";
  strategy: "structure" | "parent_child" | "semantic";
  ocr_enabled: boolean;
  extract_figures: boolean;
  extract_tables: boolean;
  target_tokens: number;
  workers: number;
  sink: "file" | "postgres";
}

export function Phase1Form({
  value,
  defaults,
  busy,
  error,
  onChange,
  onRun,
  onSkip,
  onUpload,
  onSample,
}: {
  value: Phase1State;
  defaults: Defaults | null;
  busy: boolean;
  error: string | null;
  onChange: (next: Phase1State) => void;
  onRun: () => void;
  onSkip: () => void;
  onUpload: (files: FileList) => void;
  onSample: () => void;
}) {
  const set = <K extends keyof Phase1State>(key: K, v: Phase1State[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <section className="card">
      <h2>Phase 1 — PDF extraction</h2>
      <p className="lead">
        Wraps <code>python -m phase1_pdf.pipeline</code>. Default input is the Phase 1
        repo <code>data/raw_pdfs</code>. Upload is for local-mode PDFs only.
      </p>
      <div className="grid">
        <div className="field wide">
          <label htmlFor="p1-input">PDF input path</label>
          <input
            id="p1-input"
            value={value.input_dir}
            onChange={(e) => set("input_dir", e.target.value)}
            placeholder={defaults?.phase1_input}
          />
        </div>
        <div className="field wide">
          <label htmlFor="p1-out">Output directory</label>
          <input
            id="p1-out"
            value={value.out_dir}
            onChange={(e) => set("out_dir", e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="p1-backend">Backend</label>
          <select
            id="p1-backend"
            value={value.backend}
            onChange={(e) => set("backend", e.target.value as Phase1State["backend"])}
          >
            <option value="pymupdf4llm">pymupdf4llm (fast)</option>
            <option value="docling">docling (tables)</option>
            <option value="pymupdf">pymupdf (fallback)</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="p1-strategy">Chunk strategy</label>
          <select
            id="p1-strategy"
            value={value.strategy}
            onChange={(e) => set("strategy", e.target.value as Phase1State["strategy"])}
          >
            <option value="structure">structure</option>
            <option value="parent_child">parent_child</option>
            <option value="semantic">semantic (same windows as structure today)</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="p1-tokens">Target tokens</label>
          <input
            id="p1-tokens"
            type="number"
            value={value.target_tokens}
            onChange={(e) => set("target_tokens", Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="p1-workers">Workers</label>
          <input
            id="p1-workers"
            type="number"
            value={value.workers}
            onChange={(e) => set("workers", Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="p1-sink">Sink</label>
          <select
            id="p1-sink"
            value={value.sink}
            onChange={(e) => set("sink", e.target.value as Phase1State["sink"])}
          >
            <option value="file">file (JSONL)</option>
            <option value="postgres">postgres (PG_DSN)</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="p1-upload">Upload PDFs</label>
          <input
            id="p1-upload"
            type="file"
            accept="application/pdf"
            multiple
            onChange={(e) => e.target.files && onUpload(e.target.files)}
          />
        </div>
        <div className="field wide">
          <label>Options</label>
          <div className="checks">
            <label>
              <input
                type="checkbox"
                checked={value.ocr_enabled}
                onChange={(e) => set("ocr_enabled", e.target.checked)}
              />
              OCR scanned pages
            </label>
            <label>
              <input
                type="checkbox"
                checked={value.extract_figures}
                onChange={(e) => set("extract_figures", e.target.checked)}
              />
              Figure OCR
            </label>
            <label>
              <input
                type="checkbox"
                checked={value.extract_tables}
                onChange={(e) => set("extract_tables", e.target.checked)}
              />
              Tables
            </label>
          </div>
        </div>
      </div>
      <div className="row">
        <button className="btn primary" type="button" disabled={busy} onClick={onRun}>
          Run extraction
        </button>
        <button className="btn" type="button" disabled={busy} onClick={onSample}>
          Generate sample PDF
        </button>
        <button className="btn ghost" type="button" onClick={onSkip}>
          Skip — I already have chunks
        </button>
      </div>
      {error && <div className="err">{error}</div>}
    </section>
  );
}
