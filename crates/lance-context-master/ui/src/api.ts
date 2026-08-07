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
  /**
   * Set when the master served this list from its in-memory snapshot because
   * the stats table was busy. Rows are then at most one scan interval old, and
   * a search may omit retired experiments.
   */
  stale?: boolean;
}

export interface Relationship {
  target_id: string;
  relation: string;
  weight?: number;
}

export interface RolloutRecord {
  id: string;
  rollout_id: string;
  problem_id: string;
  dataset?: string;
  sequence_order: number;
  role: string;
  created_at: string;
  content?: string;
  content_type: string;
  input_tokens?: number[];
  output_tokens?: number[];
  num_input_tokens?: number;
  num_output_tokens?: number;
  output_logprobs?: number[];
  input_logprobs?: number[];
  ref_logprobs?: number[];
  loss_mask?: number[];
  advantage?: number;
  reward?: number;
  raw_reward?: number;
  grader_id?: string;
  score?: number;
  include_in_training?: boolean;
  exclude_reason?: string;
  policy_version?: string;
  relationships?: Relationship[];
  payload_size?: number;
  payload_checksum?: string;
  artifact_type?: string;
  metadata?: unknown;
}

export interface RecordFilters {
  id: string;
  rollout_id: string;
  problem_id: string;
  dataset: string;
  role: string;
  policy_version: string;
  artifact_type: string;
  include_in_training: string;
}

export type RecordSource = "fragments" | "wal" | "all";

export interface ExperimentRecordsResponse {
  records: RolloutRecord[];
  has_more: boolean;
  limit: number;
  offset: number;
  source: RecordSource;
}

export type CompactJobStatus =
  | { state: "queued" }
  | { state: "running" }
  | { state: "done"; fragments_removed: number; fragments_added: number }
  | { state: "failed"; error: string }
  | { state: "none" };

// Unified scheduler task DTOs (mirror `TaskRecord` etc. in `lance-context-api`).
export type TaskKind = "compact" | "merge_wal" | "index_id";
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
  depends_on?: string[];
}

export interface TaskListResponse {
  tasks: TaskRecord[];
  total: number;
  limit: number;
  offset: number;
}

export interface EnqueueTaskRequest {
  kind: TaskKind;
  target: string;
  depends_on?: string[];
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

export async function listExperimentRecords(
  name: string,
  filters: RecordFilters,
  limit: number,
  offset: number,
  source: RecordSource = "fragments",
): Promise<ExperimentRecordsResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  params.set("source", source);
  for (const [key, value] of Object.entries(filters)) {
    if (value) params.set(key, value);
  }
  return json(
    await fetch(
      `${API}/experiments/${encodeURIComponent(name)}/records?${params.toString()}`,
    ),
  );
}

export function experimentBlobUrl(name: string, id: string): string {
  return `${API}/experiments/${encodeURIComponent(name)}/records/${encodeURIComponent(id)}/blob`;
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

export async function listTasks(
  limit = 50,
  offset = 0,
): Promise<TaskListResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  return json(await fetch(`${API}/tasks?${params.toString()}`));
}

export async function getTask(id: string): Promise<TaskRecord> {
  return json(await fetch(`${API}/tasks/${encodeURIComponent(id)}`));
}

export interface SqlQueryResponse {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
}

/** Run a read-only SELECT against one experiment's records (table: `records`). */
export async function runSql(
  name: string,
  sql: string,
): Promise<SqlQueryResponse> {
  return json(
    await fetch(`${API}/experiments/${encodeURIComponent(name)}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql }),
    }),
  );
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
