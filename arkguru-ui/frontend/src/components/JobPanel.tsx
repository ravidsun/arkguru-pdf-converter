import type { Job } from "../types";

export function JobPanel({
  job,
  onCancel,
}: {
  job: Job | null;
  onCancel: () => void;
}) {
  if (!job) {
    return (
      <aside className="card">
        <h2>Job log</h2>
        <p className="lead">Run a phase to stream command output here. Long OCR, crawl, and index jobs stay off the browser thread.</p>
      </aside>
    );
  }
  const running = job.status === "queued" || job.status === "running";
  return (
    <aside className="card">
      <h2>{job.kind}</h2>
      <p className="meta">
        {job.status} · {job.id}
        {job.pid ? ` · pid ${job.pid}` : ""}
        {job.error ? ` · ${job.error}` : ""}
      </p>
      <div className="logs">{job.logs.join("\n") || "waiting for output…"}</div>
      <div className="row">
        <button className="btn ghost" type="button" disabled={!running} onClick={onCancel}>
          Cancel
        </button>
      </div>
      {job.status === "succeeded" && job.result && Object.keys(job.result).length > 0 && (
        <div className="result">
          <p className="meta">Result</p>
          <pre>{JSON.stringify(job.result, null, 2)}</pre>
        </div>
      )}
    </aside>
  );
}
