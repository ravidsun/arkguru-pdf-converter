export type Step = "phase1" | "phase2" | "phase3" | "chat";

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "canceled";

export interface RepoInfo {
  found: boolean;
  path: string | null;
}

export interface Health {
  ok: boolean;
  umbrella: string;
  common: RepoInfo;
  phase1: RepoInfo;
  phase2: RepoInfo;
  phase3: RepoInfo;
  pg_dsn_set: boolean;
  ollama: boolean;
  notes: string[];
}

export interface Defaults {
  phase1_input: string;
  phase1_out: string;
  phase2_out: string;
  phase2_seeds: string[];
  phase3_store: string;
  phase3_processed: string;
  uploads: string;
  pg_dsn_set: boolean;
}

export interface Job {
  id: string;
  kind: string;
  cmd: string[];
  cwd: string;
  status: JobStatus;
  logs: string[];
  log_offset: number;
  log_total: number;
  result: Record<string, unknown>;
  error: string | null;
  returncode: number | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  pid: number | null;
}

export interface ChatSource {
  chunk_id?: string | null;
  source_id?: string | null;
  page?: number | null;
  section?: string | null;
  url?: string | null;
  chunk_index?: number | null;
  text?: string | null;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  extractive: boolean;
  ollama: boolean;
  store: string | null;
  embedder: string | null;
}

export interface ChatTurn {
  question: string;
  answer: string;
  sources: ChatSource[];
  extractive: boolean;
}

export function stepIndex(step: Step): number {
  switch (step) {
    case "phase1":
      return 0;
    case "phase2":
      return 1;
    case "phase3":
      return 2;
    case "chat":
      return 3;
    default: {
      const _exhaustive: never = step;
      return _exhaustive;
    }
  }
}

export function stepLabel(step: Step): string {
  switch (step) {
    case "phase1":
      return "1 · PDF";
    case "phase2":
      return "2 · Web";
    case "phase3":
      return "3 · Index";
    case "chat":
      return "Chat";
    default: {
      const _exhaustive: never = step;
      return _exhaustive;
    }
  }
}

export const STEPS: Step[] = ["phase1", "phase2", "phase3", "chat"];
