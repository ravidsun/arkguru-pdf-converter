import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { ChatPanel } from "./components/ChatPanel";
import { JobPanel } from "./components/JobPanel";
import { Phase1Form, type Phase1State } from "./components/Phase1Form";
import { Phase2Form, type Phase2State } from "./components/Phase2Form";
import { Phase3Form, type Phase3State } from "./components/Phase3Form";
import { StepNav } from "./components/StepNav";
import type { ChatTurn, Defaults, Health, Job, Step } from "./types";

const emptyP1 = (d?: Defaults | null): Phase1State => ({
  input_dir: d?.phase1_input ?? "",
  out_dir: d?.phase1_out ?? "",
  backend: "pymupdf4llm",
  strategy: "structure",
  ocr_enabled: true,
  extract_figures: true,
  extract_tables: true,
  target_tokens: 400,
  workers: 1,
  sink: "file",
});

const emptyP2 = (d?: Defaults | null): Phase2State => ({
  seedsText: (d?.phase2_seeds ?? []).slice(0, 2).join("\n"),
  max_pages: 6,
  max_pages_per_seed: 3,
  same_domain_only: true,
  backend: "local",
  sink: "file",
  out_dir: d?.phase2_out ?? "",
  download_pdfs: false,
});

const emptyP3 = (d?: Defaults | null): Phase3State => ({
  sink: "file",
  embedder: "hashing",
  include_phase1: true,
  include_phase2: true,
  skip_finetune: true,
  store: d?.phase3_store ?? "",
  processed_dir: d?.phase3_processed ?? "",
});

export function App() {
  const [step, setStep] = useState<Step>("phase1");
  const [health, setHealth] = useState<Health | null>(null);
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const [p1, setP1] = useState<Phase1State>(emptyP1());
  const [p2, setP2] = useState<Phase2State>(emptyP2());
  const [p3, setP3] = useState<Phase3State>(emptyP3());
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [question, setQuestion] = useState("What PPE is required before servicing a unit?");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [chatBusy, setChatBusy] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch((e: Error) => setError(e.message));
    api.defaults().then((d) => {
      setDefaults(d);
      setP1((prev) => ({ ...emptyP1(d), ...keepUser(prev) }));
      setP2((prev) => ({ ...emptyP2(d), seedsText: prev.seedsText || emptyP2(d).seedsText }));
      setP3((prev) => ({ ...emptyP3(d), store: prev.store || d.phase3_store }));
    }).catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed" || job.status === "canceled") {
      return;
    }
    const id = job.id;
    const timer = window.setInterval(() => {
      api.job(id).then(setJob).catch((e: Error) => setError(e.message));
    }, 600);
    return () => window.clearInterval(timer);
  }, [job]);

  const busy = useMemo(
    () => !!job && (job.status === "queued" || job.status === "running"),
    [job],
  );

  const watch = async (started: Job) => {
    setError(null);
    setJob(started);
  };

  const runPhase1 = async () => {
    try {
      const { job: started } = await api.startPhase1({ ...p1 });
      await watch(started);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const runSample = async () => {
    try {
      const { job: started } = await api.samplePdf();
      await watch(started);
      const done = await waitFor(started.id);
      setJob(done);
      const input = done.result?.input_dir;
      if (typeof input === "string") {
        setP1((prev) => ({ ...prev, input_dir: input }));
      }
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const upload = async (files: FileList) => {
    try {
      const res = await api.upload(files);
      setP1((prev) => ({ ...prev, input_dir: res.dir }));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const runPhase2 = async () => {
    try {
      const seeds = p2.seedsText.split(/\n+/).map((s) => s.trim()).filter(Boolean);
      const { job: started } = await api.startPhase2({
        seeds,
        max_pages: p2.max_pages,
        max_pages_per_seed: p2.max_pages_per_seed,
        same_domain_only: p2.same_domain_only,
        backend: p2.backend,
        sink: p2.sink,
        out_dir: p2.out_dir,
        download_pdfs: p2.download_pdfs,
      });
      await watch(started);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const runPhase3 = async () => {
    try {
      const { job: started } = await api.startPhase3({ ...p3, skip_finetune: true });
      await watch(started);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const sendChat = async () => {
    setChatBusy(true);
    setError(null);
    try {
      const res = await api.chat({
        question,
        embedder: p3.embedder,
        sink: p3.sink,
        store: p3.store,
      });
      setTurns((prev) => [
        ...prev,
        { question, answer: res.answer, sources: res.sources, extractive: res.extractive },
      ]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setChatBusy(false);
    }
  };

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <h1>arkguru <span>local RAG wizard</span></h1>
          <p>
            Guided wrapper around the existing Phase 1 / 2 / 3 CLIs. Nothing here
            reimplements extraction, crawl, or retrieval — it runs the packages
            already in this monorepo and streams their logs.
          </p>
        </div>
        <div className="pills">
          <span className={`pill ${health?.phase1.found ? "ok" : "bad"}`}>
            phase1 {health?.phase1.found ? "found" : "missing"}
          </span>
          <span className={`pill ${health?.phase2.found ? "ok" : "bad"}`}>
            phase2 {health?.phase2.found ? "found" : "missing"}
          </span>
          <span className={`pill ${health?.phase3.found ? "ok" : "bad"}`}>
            phase3 {health?.phase3.found ? "found" : "missing"}
          </span>
          <span className={`pill ${health?.pg_dsn_set ? "ok" : ""}`}>
            PG_DSN {health?.pg_dsn_set ? "set" : "unset"}
          </span>
          <span className={`pill ${health?.ollama ? "ok" : ""}`}>
            ollama {health?.ollama ? "up" : "off"}
          </span>
        </div>
      </header>

      <StepNav step={step} onChange={setStep} />

      <div className="layout">
        {step === "phase1" && (
          <Phase1Form
            value={p1}
            defaults={defaults}
            busy={busy}
            error={error}
            onChange={setP1}
            onRun={runPhase1}
            onSkip={() => setStep("phase2")}
            onUpload={upload}
            onSample={runSample}
          />
        )}
        {step === "phase2" && (
          <Phase2Form
            value={p2}
            defaults={defaults}
            busy={busy}
            error={error}
            onChange={setP2}
            onRun={runPhase2}
            onSkip={() => setStep("phase3")}
            onLoadSeeds={() =>
              setP2((prev) => ({ ...prev, seedsText: (defaults?.phase2_seeds ?? []).join("\n") }))
            }
          />
        )}
        {step === "phase3" && (
          <Phase3Form
            value={p3}
            pgDsnSet={!!health?.pg_dsn_set}
            busy={busy}
            error={error}
            onChange={setP3}
            onRun={runPhase3}
            onSkip={() => setStep("chat")}
          />
        )}
        {step === "chat" && (
          <ChatPanel
            turns={turns}
            question={question}
            busy={chatBusy}
            error={error}
            extractiveHint={!health?.ollama}
            onQuestion={setQuestion}
            onSend={sendChat}
          />
        )}
        <JobPanel
          job={job}
          onCancel={() => job && api.cancel(job.id).then((r) => setJob(r.job))}
        />
      </div>
    </div>
  );
}

function keepUser(prev: Phase1State): Partial<Phase1State> {
  return prev.input_dir ? { input_dir: prev.input_dir } : {};
}

async function waitFor(id: string): Promise<Job> {
  for (let i = 0; i < 80; i += 1) {
    const current = await api.job(id);
    if (current.status === "succeeded" || current.status === "failed" || current.status === "canceled") {
      return current;
    }
    await new Promise((r) => window.setTimeout(r, 400));
  }
  return api.job(id);
}
