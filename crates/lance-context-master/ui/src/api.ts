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

export interface ExperimentRecordsResponse {
  records: RolloutRecord[];
  has_more: boolean;
  limit: number;
  offset: number;
}

export type CompactJobStatus =
  | { state: "queued" }
  | { state: "running" }
  | { state: "done"; fragments_removed: number; fragments_added: number }
  | { state: "failed"; error: string }
  | { state: "none" };

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
): Promise<ExperimentRecordsResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  params.set("offset", String(offset));
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
