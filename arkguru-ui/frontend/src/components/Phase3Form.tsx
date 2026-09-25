export interface Phase3State {
  sink: "file" | "postgres";
  embedder: "hashing" | "sentence_transformer";
  include_phase1: boolean;
  include_phase2: boolean;
  skip_finetune: boolean;
  store: string;
  processed_dir: string;
}

export function Phase3Form({
  value,
  pgDsnSet,
  busy,
  error,
  onChange,
  onRun,
  onSkip,
}: {
  value: Phase3State;
  pgDsnSet: boolean;
  busy: boolean;
  error: string | null;
  onChange: (next: Phase3State) => void;
  onRun: () => void;
  onSkip: () => void;
}) {
  const set = <K extends keyof Phase3State>(key: K, v: Phase3State[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <section className="card">
      <h2>Phase 3 — Combine, index, skip fine-tune</h2>
      <p className="lead">
        File mode calls <code>phase3_rag.prepare_dataset</code> then
        <code> phase3_rag.quickstart --add</code>. Postgres mode calls
        <code> phase3_rag.embed_datastore</code>. Fine-tune stays CLI-only.
      </p>
      <div className="banner">
        Fine-tune is skipped in this wizard. To train LoRA: from
        <code> arkguru-rag-slm</code> run <code>make finetune</code>, then merge / GGUF /
        <code>ollama create</code> as documented in <code>phase3_rag/serve.py</code>.
      </div>
      <div className="grid">
        <div className="field">
          <label htmlFor="p3-sink">Index sink</label>
          <select
            id="p3-sink"
            value={value.sink}
            onChange={(e) => set("sink", e.target.value as Phase3State["sink"])}
          >
            <option value="file">file (LocalVectorStore)</option>
            <option value="postgres" disabled={!pgDsnSet}>
              postgres {pgDsnSet ? "" : "(PG_DSN not set)"}
            </option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="p3-emb">Embedder</label>
          <select
            id="p3-emb"
            value={value.embedder}
            onChange={(e) => set("embedder", e.target.value as Phase3State["embedder"])}
          >
            <option value="hashing">hashing (smoke, no download)</option>
            <option value="sentence_transformer">sentence_transformer (prod)</option>
          </select>
        </div>
        <div className="field wide">
          <label htmlFor="p3-store">Store path</label>
          <input
            id="p3-store"
            value={value.store}
            onChange={(e) => set("store", e.target.value)}
          />
        </div>
        <div className="field wide">
          <label htmlFor="p3-proc">Processed dir (corpus output)</label>
          <input
            id="p3-proc"
            value={value.processed_dir}
            onChange={(e) => set("processed_dir", e.target.value)}
          />
        </div>
        <div className="field wide">
          <label>Sources</label>
          <div className="checks">
            <label>
              <input
                type="checkbox"
                checked={value.include_phase1}
                onChange={(e) => set("include_phase1", e.target.checked)}
              />
              Include Phase 1 JSONL
            </label>
            <label>
              <input
                type="checkbox"
                checked={value.include_phase2}
                onChange={(e) => set("include_phase2", e.target.checked)}
              />
              Include Phase 2 JSONL
            </label>
            <label>
              <input type="checkbox" checked={value.skip_finetune} readOnly />
              Skip fine-tune (CLI-only)
            </label>
          </div>
        </div>
      </div>
      <div className="row">
        <button className="btn primary" type="button" disabled={busy} onClick={onRun}>
          Combine + index
        </button>
        <button className="btn ghost" type="button" onClick={onSkip}>
          Skip to chat
        </button>
      </div>
      {error && <div className="err">{error}</div>}
    </section>
  );
}
