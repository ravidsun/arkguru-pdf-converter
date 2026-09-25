import type { Defaults } from "../types";

export interface Phase2State {
  seedsText: string;
  max_pages: number;
  max_pages_per_seed: number;
  same_domain_only: boolean;
  backend: "local" | "firecrawl";
  sink: "file" | "postgres";
  out_dir: string;
  download_pdfs: boolean;
}

export function Phase2Form({
  value,
  defaults,
  busy,
  error,
  onChange,
  onRun,
  onSkip,
  onLoadSeeds,
}: {
  value: Phase2State;
  defaults: Defaults | null;
  busy: boolean;
  error: string | null;
  onChange: (next: Phase2State) => void;
  onRun: () => void;
  onSkip: () => void;
  onLoadSeeds: () => void;
}) {
  const set = <K extends keyof Phase2State>(key: K, v: Phase2State[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <section className="card">
      <h2>Phase 2 — Web scraping</h2>
      <p className="lead">
        Wraps <code>python -m phase2_web.pipeline</code> with a temp YAML so
        <code> same_domain_only</code> and <code>backend</code> reach <code>Phase2Config</code>.
        The current crawl implementation is local httpx + trafilatura.
      </p>
      <div className="grid">
        <div className="field wide">
          <label htmlFor="p2-seeds">Seed URLs (one per line)</label>
          <textarea
            id="p2-seeds"
            value={value.seedsText}
            onChange={(e) => set("seedsText", e.target.value)}
            placeholder={defaults?.phase2_seeds.join("\n")}
          />
        </div>
        <div className="field">
          <label htmlFor="p2-max">max_pages</label>
          <input
            id="p2-max"
            type="number"
            value={value.max_pages}
            onChange={(e) => set("max_pages", Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="p2-per">max_pages_per_seed</label>
          <input
            id="p2-per"
            type="number"
            value={value.max_pages_per_seed}
            onChange={(e) => set("max_pages_per_seed", Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="p2-backend">Backend</label>
          <select
            id="p2-backend"
            value={value.backend}
            onChange={(e) => set("backend", e.target.value as Phase2State["backend"])}
          >
            <option value="local">local</option>
            <option value="firecrawl">firecrawl (config only — not wired in crawl())</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="p2-sink">Sink</label>
          <select
            id="p2-sink"
            value={value.sink}
            onChange={(e) => set("sink", e.target.value as Phase2State["sink"])}
          >
            <option value="file">file (web_chunks.jsonl)</option>
            <option value="postgres">postgres schema web</option>
          </select>
        </div>
        <div className="field wide">
          <label htmlFor="p2-out">Output directory</label>
          <input
            id="p2-out"
            value={value.out_dir}
            onChange={(e) => set("out_dir", e.target.value)}
          />
        </div>
        <div className="field wide">
          <label>Options</label>
          <div className="checks">
            <label>
              <input
                type="checkbox"
                checked={value.same_domain_only}
                onChange={(e) => set("same_domain_only", e.target.checked)}
              />
              same_domain_only
            </label>
            <label>
              <input
                type="checkbox"
                checked={value.download_pdfs}
                onChange={(e) => set("download_pdfs", e.target.checked)}
              />
              Download linked PDFs
            </label>
          </div>
        </div>
      </div>
      <div className="row">
        <button className="btn primary" type="button" disabled={busy} onClick={onRun}>
          Run crawl
        </button>
        <button className="btn" type="button" onClick={onLoadSeeds}>
          Load default seeds
        </button>
        <button className="btn ghost" type="button" onClick={onSkip}>
          Skip Phase 2
        </button>
      </div>
      {value.backend === "firecrawl" && (
        <div className="banner">
          firecrawl is accepted on the config object. <code>phase2_web.fetch.crawl</code> still
          uses the local backend. Keep <code>FIRECRAWL_API_KEY</code> in <code>.env</code> if you
          later wire it.
        </div>
      )}
      {error && <div className="err">{error}</div>}
    </section>
  );
}
