import type { ChatResponse, Defaults, Health, Job } from "./types";

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail || body);
  } catch {
    return res.statusText;
  }
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<Health>(r)),
  defaults: () => fetch("/api/defaults").then((r) => json<Defaults>(r)),
  corpus: () => fetch("/api/corpus").then((r) => json<Record<string, unknown>>(r)),

  startPhase1: (body: Record<string, unknown>) =>
    fetch("/api/jobs/phase1", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<{ job: Job }>(r)),

  startPhase2: (body: Record<string, unknown>) =>
    fetch("/api/jobs/phase2", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<{ job: Job }>(r)),

  startPhase3: (body: Record<string, unknown>) =>
    fetch("/api/jobs/phase3", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<{ job: Job }>(r)),

  samplePdf: () =>
    fetch("/api/jobs/sample-pdf", { method: "POST" }).then((r) =>
      json<{ job: Job }>(r),
    ),

  job: (id: string, logOffset = 0) =>
    fetch(`/api/jobs/${id}?log_offset=${logOffset}`).then((r) => json<Job>(r)),

  cancel: (id: string) =>
    fetch(`/api/jobs/${id}/cancel`, { method: "POST" }).then((r) =>
      json<{ job: Job }>(r),
    ),

  upload: async (files: FileList | File[]) => {
    const data = new FormData();
    Array.from(files).forEach((f) => data.append("files", f));
    const res = await fetch("/api/uploads", { method: "POST", body: data });
    return json<{ dir: string; files: string[] }>(res);
  },

  chat: (body: Record<string, unknown>) =>
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<ChatResponse>(r)),
};
