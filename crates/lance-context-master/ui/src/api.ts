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
