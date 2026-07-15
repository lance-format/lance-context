// Wire types + fetch helpers for the master admin API. These mirror the Rust
// DTOs in `lance-context-api` (ExperimentSummary / CompactJobStatus).

export interface ExperimentSummary {
  name: string;
  uri: string;
  row_count: number;
  fragment_count: number;
  last_updated: number;
  pending_wal_generations: number;
  last_compaction?: number | null;
  total_compactions: number;
  scanned_at: number;
}

export interface ExperimentListResponse {
  experiments: ExperimentSummary[];
  total: number;
}

export type CompactJobStatus =
  | { state: "queued" }
  | { state: "running" }
  | { state: "done"; fragments_removed: number; fragments_added: number }
  | { state: "failed"; error: string }
  | { state: "none" };

// Unified scheduler task DTOs (mirror `TaskRecord` etc. in `lance-context-api`).
export type TaskKind = "compact" | "merge_wal";
export type TaskState = "queued" | "running" | "done" | "failed";

export interface TaskRecord {
  id: string;
  kind: TaskKind;
  target: string;
  state: TaskState;
  error?: string | null;
  detail?: string | null;
  enqueued_at: number;
  started_at?: number | null;
  finished_at?: number | null;
}

export interface TaskListResponse {
  tasks: TaskRecord[];
}

export interface EnqueueTaskRequest {
  kind: TaskKind;
  target: string;
}

const API = "/api/v1";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function listExperiments(
  search: string,
  limit: number,
  offset: number,
): Promise<ExperimentListResponse> {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  return json(await fetch(`${API}/experiments?${params.toString()}`));
}

export async function getExperiment(
  name: string,
  fresh = false,
): Promise<{ summary?: ExperimentSummary } & ExperimentSummary> {
  const q = fresh ? "?fresh=true" : "";
  return json(await fetch(`${API}/experiments/${encodeURIComponent(name)}${q}`));
}

export async function triggerCompaction(name: string): Promise<CompactJobStatus> {
  return json(
    await fetch(`${API}/experiments/${encodeURIComponent(name)}/compact`, {
      method: "POST",
    }),
  );
}

export async function compactionStatus(name: string): Promise<CompactJobStatus> {
  return json(
    await fetch(`${API}/experiments/${encodeURIComponent(name)}/compact/status`),
  );
}

export async function listTasks(): Promise<TaskListResponse> {
  return json(await fetch(`${API}/tasks`));
}

export async function getTask(id: string): Promise<TaskRecord> {
  return json(await fetch(`${API}/tasks/${encodeURIComponent(id)}`));
}

export async function enqueueTask(req: EnqueueTaskRequest): Promise<TaskRecord> {
  return json(
    await fetch(`${API}/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  );
}
