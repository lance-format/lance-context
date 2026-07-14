import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  compactionStatus,
  getExperiment,
  listExperiments,
  triggerCompaction,
  type CompactJobStatus,
  type ExperimentSummary,
} from "./api";
import { useUiStore } from "./store";

function fmtTime(ms: number | null | undefined): string {
  if (!ms || ms < 0) return "—";
  return new Date(ms).toLocaleString();
}

function statusLabel(s: CompactJobStatus | undefined): string {
  if (!s) return "";
  switch (s.state) {
    case "queued":
      return "queued";
    case "running":
      return "running…";
    case "done":
      return `done (-${s.fragments_removed}/+${s.fragments_added})`;
    case "failed":
      return `failed: ${s.error}`;
    case "none":
      return "";
  }
}

function CompactButton({ name }: { name: string }) {
  const qc = useQueryClient();
  const [poll, setPoll] = useState(false);
  const status = useQuery({
    queryKey: ["compact-status", name],
    queryFn: () => compactionStatus(name),
    refetchInterval: poll ? 1000 : false,
  });
  const trigger = useMutation({
    mutationFn: () => triggerCompaction(name),
    onSuccess: () => {
      setPoll(true);
      qc.invalidateQueries({ queryKey: ["compact-status", name] });
    },
  });

  const s = status.data;
  // Stop polling once we reach a terminal state and refresh the list metrics.
  if (poll && (s?.state === "done" || s?.state === "failed")) {
    setPoll(false);
    qc.invalidateQueries({ queryKey: ["experiments"] });
  }

  return (
    <span>
      <button
        onClick={() => trigger.mutate()}
        disabled={trigger.isPending || s?.state === "running" || s?.state === "queued"}
      >
        Compact
      </button>
      <span style={{ marginLeft: 8, color: "#666", fontSize: 12 }}>
        {statusLabel(s)}
      </span>
    </span>
  );
}

function ExperimentRow({ exp }: { exp: ExperimentSummary }) {
  const select = useUiStore((s) => s.select);
  return (
    <tr>
      <td>
        <a href="#" onClick={(e) => (e.preventDefault(), select(exp.name))}>
          {exp.name}
        </a>
      </td>
      <td style={{ textAlign: "right" }}>{exp.row_count.toLocaleString()}</td>
      <td style={{ textAlign: "right" }}>{exp.fragment_count.toLocaleString()}</td>
      <td style={{ textAlign: "right" }}>{exp.pending_wal_generations}</td>
      <td>{fmtTime(exp.last_updated)}</td>
      <td>{fmtTime(exp.last_compaction)}</td>
      <td>
        <CompactButton name={exp.name} />
      </td>
    </tr>
  );
}

function DetailPanel({ name }: { name: string }) {
  const select = useUiStore((s) => s.select);
  const detail = useQuery({
    queryKey: ["experiment", name],
    queryFn: () => getExperiment(name, true),
  });
  const d = detail.data;
  return (
    <div style={{ border: "1px solid #ddd", padding: 16, marginBottom: 16 }}>
      <button onClick={() => select(null)}>← back</button>
      <h2>{name}</h2>
      {detail.isLoading && <p>loading…</p>}
      {d && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <Metric label="Rows" value={d.row_count.toLocaleString()} />
          <Metric label="Fragments" value={d.fragment_count.toLocaleString()} />
          <Metric label="Pending WAL gens" value={String(d.pending_wal_generations)} />
          <Metric label="Total compactions" value={String(d.total_compactions)} />
          <Metric label="Last updated" value={fmtTime(d.last_updated)} />
          <Metric label="Last compaction" value={fmtTime(d.last_compaction)} />
          <Metric label="URI" value={d.uri} />
          <Metric label="Scanned at" value={fmtTime(d.scanned_at)} />
        </div>
      )}
      <div style={{ marginTop: 12 }}>
        <CompactButton name={name} />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: 12, color: "#888" }}>{label}</div>
      <div style={{ fontFamily: "monospace" }}>{value}</div>
    </div>
  );
}

export function App() {
  const { search, page, pageSize, selected, setSearch, setPage } = useUiStore();
  const list = useQuery({
    queryKey: ["experiments", search, page, pageSize],
    queryFn: () => listExperiments(search, pageSize, page * pageSize),
  });

  const total = list.data?.total ?? 0;
  const maxPage = Math.max(0, Math.ceil(total / pageSize) - 1);

  return (
    <div style={{ maxWidth: 1100, margin: "24px auto", fontFamily: "sans-serif" }}>
      <h1>lance-context master</h1>

      {selected && <DetailPanel name={selected} />}

      <div style={{ marginBottom: 12 }}>
        <input
          placeholder="search experiment_name…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ padding: 6, width: 320 }}
        />
        <span style={{ marginLeft: 12, color: "#666" }}>{total} experiments</span>
      </div>

      {list.isLoading && <p>loading…</p>}
      {list.isError && <p style={{ color: "red" }}>{String(list.error)}</p>}

      {list.data && (
        <table style={{ width: "100%", borderCollapse: "collapse" }} border={1} cellPadding={6}>
          <thead>
            <tr>
              <th>name</th>
              <th>rows</th>
              <th>fragments</th>
              <th>pending</th>
              <th>last updated</th>
              <th>last compaction</th>
              <th>action</th>
            </tr>
          </thead>
          <tbody>
            {list.data.experiments.map((exp) => (
              <ExperimentRow key={exp.name} exp={exp} />
            ))}
          </tbody>
        </table>
      )}

      <div style={{ marginTop: 12 }}>
        <button disabled={page <= 0} onClick={() => setPage(page - 1)}>
          prev
        </button>
        <span style={{ margin: "0 12px" }}>
          page {page + 1} / {maxPage + 1}
        </span>
        <button disabled={page >= maxPage} onClick={() => setPage(page + 1)}>
          next
        </button>
      </div>
    </div>
  );
}
