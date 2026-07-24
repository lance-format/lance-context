import { useMutation, useQuery, useQueryClient, useIsFetching } from "@tanstack/react-query";
import {
  Fragment,
  useEffect,
  useState,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import {
  NavLink,
  Navigate,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  compactionStatus,
  enqueueTask,
  experimentBlobUrl,
  getExperiment,
  listExperimentRecords,
  listExperiments,
  listTasks,
  runSql,
  triggerCompaction,
  type CompactJobStatus,
  type ExperimentSummary,
  type RecordFilters,
  type RecordSource,
  type RolloutRecord,
  type SqlQueryResponse,
  type TaskRecord,
  type TaskState,
} from "./api";
import { useUiStore } from "./store";

/* ---- formatting helpers -------------------------------------------------- */

function fmtInt(n: number): string {
  return n.toLocaleString("en-US");
}

function fmtCompact(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, "") + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
  return String(n);
}

function fmtTime(ms: number | null | undefined): string {
  if (!ms || ms < 0) return "—";
  return new Date(ms).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function relTime(ms: number | null | undefined): string {
  if (!ms || ms < 0) return "never";
  const d = Date.now() - ms;
  const s = Math.floor(d / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/* ---- icons --------------------------------------------------------------- */

function SearchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" strokeLinecap="round" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M18 6 6 18M6 6l12 12" strokeLinecap="round" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 3v12m0 0 5-5m-5 5-5-5M5 21h14" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function LinkIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path
        d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Copy an absolute URL (origin + given path) to the clipboard. Best-effort:
 * silently ignores clipboard failures (e.g. non-secure contexts). */
function copyLink(path: string) {
  const url = `${window.location.origin}${path}`;
  void navigator.clipboard?.writeText(url).catch(() => {});
}

function ChevronIcon({ open }: { open: boolean }) {  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      className={open ? "chevron chevron--open" : "chevron"}
    >
      <path d="m9 18 6-6-6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/* ---- compaction status --------------------------------------------------- */

function StatusPill({ status }: { status: CompactJobStatus | undefined }) {
  if (!status || status.state === "none") return null;
  switch (status.state) {
    case "queued":
      return (
        <span className="pill pill--queued">
          <span className="pill__dot" />
          queued
        </span>
      );
    case "running":
      return (
        <span className="pill pill--running">
          <span className="pill__dot" />
          running
        </span>
      );
    case "done":
      return (
        <span className="pill pill--done" title="fragments removed / added">
          <span className="pill__dot" />−{status.fragments_removed} / +{status.fragments_added}
        </span>
      );
    case "failed":
      return (
        <span className="pill pill--failed" title={status.error}>
          <span className="pill__dot" />
          failed
        </span>
      );
  }
}

/** Trigger + live status for one experiment's compaction job. */
function useCompaction(name: string) {
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
  useEffect(() => {
    if (poll && (s?.state === "done" || s?.state === "failed")) {
      setPoll(false);
      qc.invalidateQueries({ queryKey: ["experiments"] });
      qc.invalidateQueries({ queryKey: ["experiment", name] });
    }
  }, [poll, s?.state, qc, name]);

  const busy = trigger.isPending || s?.state === "running" || s?.state === "queued";
  return { status: s, trigger, busy };
}

/** "Optimize" trigger — enqueues the full maintenance sequence for one
 * experiment: merge WAL into the base table, compact its fragments, then build
 * the ZoneMap index on `id`. Enqueued in that order so the scheduler drains
 * them merge → compact → index. Used in the experiment list; the detail drawer
 * keeps the three individual buttons. */
function OptimizeButton({ name }: { name: string }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const trigger = useMutation({
    mutationFn: async () => {
      // Chain the three tasks so they run in order: compact waits for the WAL
      // merge, and the id-index build waits for compaction. compact and
      // index_id share a per-experiment base-table write gate, so running them
      // concurrently would make one fail; the dependency chain serializes them.
      const merge = await enqueueTask({ kind: "merge_wal", target: name });
      const compact = await enqueueTask({
        kind: "compact",
        target: name,
        depends_on: [merge.id],
      });
      await enqueueTask({
        kind: "index_id",
        target: name,
        depends_on: [compact.id],
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tasks"] });
      navigate("/tasks");
    },
  });
  return (
    <button
      className="btn"
      onClick={() => trigger.mutate()}
      disabled={trigger.isPending}
    >
      {trigger.isPending ? "Optimizing…" : "Optimize"}
    </button>
  );
}

function CompactButton({ name, variant }: { name: string; variant?: "accent" }) {
  const { status, trigger, busy } = useCompaction(name);
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
      <button
        className={`btn ${variant === "accent" ? "btn--accent" : ""}`}
        onClick={() => trigger.mutate()}
        disabled={busy}
      >
        {busy ? "Compacting…" : "Compact"}
      </button>
      <StatusPill status={status} />
    </span>
  );
}

/* ---- table --------------------------------------------------------------- */

function Row({ exp }: { exp: ExperimentSummary }) {
  const navigate = useNavigate();
  const hot = exp.fragment_count >= 16;
  return (
    <tr>
      <td>
        <button
          className="name-cell"
          onClick={() => navigate(`/experiments/${encodeURIComponent(exp.name)}`)}
        >
          {exp.name}
        </button>
      </td>
      <td className="num">{fmtInt(exp.row_count)}</td>
      <td className={`num ${hot ? "frag--hot" : ""}`}>{fmtInt(exp.fragment_count)}</td>
      <td className="num">{exp.pending_wal_generations}</td>
      <td className="muted">{relTime(exp.last_updated)}</td>
      <td className="muted">{relTime(exp.last_compaction)}</td>
      <td style={{ textAlign: "right" }}>
        <OptimizeButton name={exp.name} />
      </td>
    </tr>
  );
}

/* ---- detail drawer ------------------------------------------------------- */

function Metric({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={`metric ${wide ? "metric--wide" : ""}`}>
      <div className="metric__label">{label}</div>
      <div className="metric__value">{value}</div>
    </div>
  );
}

const EMPTY_RECORD_FILTERS: RecordFilters = {
  id: "",
  rollout_id: "",
  problem_id: "",
  dataset: "",
  role: "",
  policy_version: "",
  artifact_type: "",
  include_in_training: "",
};

function fmtBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function preview(content: string | undefined): string {
  if (!content) return "—";
  return content.length > 120 ? `${content.slice(0, 117)}…` : content;
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

function RecordDetails({ record, name }: { record: RolloutRecord; name: string }) {
  return (
    <div className="record-detail">
      <div className="record-detail__toolbar">
        <button
          className="btn btn--ghost btn--sm"
          onClick={() =>
            copyLink(
              `/experiments/${encodeURIComponent(name)}/records/${encodeURIComponent(record.id)}`,
            )
          }
        >
          <LinkIcon />
          Copy link
        </button>
      </div>
      <div className="record-detail__grid">
        <div>
          <span>ID</span>
          <code>{record.id}</code>
        </div>
        <div>
          <span>Rollout ID</span>
          <code>{record.rollout_id}</code>
        </div>
        <div>
          <span>Problem ID</span>
          <code>{record.problem_id}</code>
        </div>
        <div>
          <span>Dataset</span>
          <code>{record.dataset ?? "—"}</code>
        </div>
        <div>
          <span>Content type</span>
          <code>{record.content_type}</code>
        </div>
        <div>
          <span>Sequence</span>
          <code>{record.sequence_order}</code>
        </div>
        <div>
          <span>Input tokens</span>
          <code>{record.num_input_tokens ?? record.input_tokens?.length ?? "—"}</code>
        </div>
        <div>
          <span>Output tokens</span>
          <code>{record.num_output_tokens ?? record.output_tokens?.length ?? "—"}</code>
        </div>
        <div>
          <span>Training</span>
          <code>
            {record.include_in_training == null
              ? "—"
              : record.include_in_training
                ? "included"
                : "excluded"}
          </code>
        </div>
        <div>
          <span>Policy</span>
          <code>{record.policy_version ?? "—"}</code>
        </div>
        <div>
          <span>Artifact type</span>
          <code>{record.artifact_type ?? "—"}</code>
        </div>
        <div>
          <span>Checksum</span>
          <code>{record.payload_checksum ?? "—"}</code>
        </div>
      </div>

      {record.content && (
        <div className="record-detail__section">
          <span>Content</span>
          <pre>{record.content}</pre>
        </div>
      )}
      {record.exclude_reason && (
        <div className="record-detail__section">
          <span>Exclude reason</span>
          <p>{record.exclude_reason}</p>
        </div>
      )}
      {record.metadata != null && (
        <div className="record-detail__section">
          <span>Metadata</span>
          <JsonBlock value={record.metadata} />
        </div>
      )}
      {(record.relationships?.length ?? 0) > 0 && (
        <div className="record-detail__section">
          <span>Relationships</span>
          <JsonBlock value={record.relationships} />
        </div>
      )}
      {(record.input_tokens || record.output_tokens || record.loss_mask) && (
        <div className="record-detail__section">
          <span>Token data</span>
          <JsonBlock
            value={{
              input_tokens: record.input_tokens,
              output_tokens: record.output_tokens,
              loss_mask: record.loss_mask,
              output_logprobs: record.output_logprobs,
              input_logprobs: record.input_logprobs,
              ref_logprobs: record.ref_logprobs,
            }}
          />
        </div>
      )}
      {record.payload_size != null && (
        <a className="btn btn--accent blob-download" href={experimentBlobUrl(name, record.id)} download>
          <DownloadIcon />
          Download blob ({fmtBytes(record.payload_size)})
        </a>
      )}
    </div>
  );
}

function RecordsView({
  name,
  expandedId,
  onToggle,
}: {
  name: string;
  expandedId: string | null;
  onToggle: (id: string | null) => void;
}) {
  const [draft, setDraft] = useState<RecordFilters>({ ...EMPTY_RECORD_FILTERS });
  const [filters, setFilters] = useState<RecordFilters>({ ...EMPTY_RECORD_FILTERS });
  const [source, setSource] = useState<RecordSource>("fragments");
  const [page, setPage] = useState(0);
  const pageSize = 25;
  const records = useQuery({
    queryKey: ["experiment-records", name, filters, source, page],
    queryFn: () =>
      listExperimentRecords(name, filters, pageSize, page * pageSize, source),
    placeholderData: (previous) => previous,
  });
  const rows = records.data?.records ?? [];
  const hasMore = records.data?.has_more ?? false;

  const selectSource = (next: RecordSource) => {
    setSource(next);
    setPage(0);
    onToggle(null);
  };
  const setFilter = (key: keyof RecordFilters, value: string) => {
    setDraft((current) => ({ ...current, [key]: value }));
  };
  const apply = (event: FormEvent) => {
    event.preventDefault();
    setFilters({ ...draft });
    setPage(0);
    onToggle(null);
  };
  const clear = () => {
    setDraft({ ...EMPTY_RECORD_FILTERS });
    setFilters({ ...EMPTY_RECORD_FILTERS });
    setPage(0);
    onToggle(null);
  };
  const hasFilters = Object.values(filters).some(Boolean);

  const sourceTabs: { key: RecordSource; label: string; title: string }[] = [
    {
      key: "fragments",
      label: "Fragments",
      title:
        "Base table only. Fast and independent of WAL backlog, but may lag the most recent (un-merged) writes.",
    },
    {
      key: "wal",
      label: "WAL",
      title: "Pending MemWAL generations only — the not-yet-merged tail.",
    },
    {
      key: "all",
      label: "All",
      title: "Base table unioned with pending WAL (fully consistent).",
    },
  ];

  return (
    <div className="records-view">
      <div className="records-source" role="tablist" aria-label="Record source">
        {sourceTabs.map((t) => (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={source === t.key}
            title={t.title}
            className={
              source === t.key
                ? "records-source__tab records-source__tab--active"
                : "records-source__tab"
            }
            onClick={() => selectSource(t.key)}
          >
            {t.label}
          </button>
        ))}
        {source === "fragments" && (
          <span className="records-source__note">
            base table only — may lag un-merged writes
          </span>
        )}
      </div>
      <form className="record-filters" onSubmit={apply}>
        <label>
          <span>ID</span>
          <input value={draft.id} onChange={(event) => setFilter("id", event.target.value)} />
        </label>
        <label>
          <span>Rollout ID</span>
          <input
            value={draft.rollout_id}
            onChange={(event) => setFilter("rollout_id", event.target.value)}
          />
        </label>
        <label>
          <span>Problem ID</span>
          <input
            value={draft.problem_id}
            onChange={(event) => setFilter("problem_id", event.target.value)}
          />
        </label>
        <label>
          <span>Dataset</span>
          <input
            value={draft.dataset}
            onChange={(event) => setFilter("dataset", event.target.value)}
          />
        </label>
        <label>
          <span>Role</span>
          <select value={draft.role} onChange={(event) => setFilter("role", event.target.value)}>
            <option value="">Any role</option>
            <option value="assistant">assistant</option>
            <option value="tool">tool</option>
            <option value="grade">grade</option>
            <option value="artifact">artifact</option>
            <option value="user">user</option>
            <option value="system">system</option>
          </select>
        </label>
        <label>
          <span>Policy version</span>
          <input
            value={draft.policy_version}
            onChange={(event) => setFilter("policy_version", event.target.value)}
          />
        </label>
        <label>
          <span>Artifact type</span>
          <input
            value={draft.artifact_type}
            onChange={(event) => setFilter("artifact_type", event.target.value)}
          />
        </label>
        <label>
          <span>Training</span>
          <select
            value={draft.include_in_training}
            onChange={(event) => setFilter("include_in_training", event.target.value)}
          >
            <option value="">Any</option>
            <option value="true">Included</option>
            <option value="false">Excluded</option>
          </select>
        </label>
        <div className="record-filters__actions">
          <button className="btn btn--accent" type="submit">
            Apply
          </button>
          <button className="btn btn--ghost" type="button" onClick={clear}>
            Clear
          </button>
        </div>
      </form>

      <div className="records-summary">
        <span>
          {rows.length === 0
            ? "no records"
            : `records ${fmtInt(page * pageSize + 1)}–${fmtInt(page * pageSize + rows.length)}`}
        </span>
        {hasFilters && <span className="filter-state">filtered</span>}
        {records.isFetching && <span className="records-sync">syncing</span>}
      </div>

      <div className="records-table-wrap">
        {records.isError && <div className="error">{String(records.error)}</div>}
        {!records.isError && (
          <table className="records-table">
            <thead>
              <tr>
                <th aria-label="expand" />
                <th>ID</th>
                <th>Rollout</th>
                <th>Role</th>
                <th>Content</th>
                <th className="num">Reward / score</th>
                <th>Policy</th>
                <th>Created</th>
                <th>Blob</th>
              </tr>
            </thead>
            <tbody>
              {records.isLoading &&
                Array.from({ length: 6 }).map((_, index) => (
                  <tr key={index}>
                    {Array.from({ length: 9 }).map((__, cell) => (
                      <td key={cell}>
                        <div className="skeleton" />
                      </td>
                    ))}
                  </tr>
                ))}
              {!records.isLoading &&
                rows.map((record) => {
                  const open = expandedId === record.id;
                  const signal = record.score ?? record.reward;
                  return (
                    <Fragment key={record.id}>
                      <tr className={open ? "record-row record-row--open" : "record-row"}>
                        <td>
                          <button
                            className="record-expand"
                            onClick={() => onToggle(open ? null : record.id)}
                            aria-label={open ? "collapse record" : "expand record"}
                          >
                            <ChevronIcon open={open} />
                          </button>
                        </td>
                        <td>
                          <button className="record-id" onClick={() => onToggle(open ? null : record.id)}>
                            {record.id}
                          </button>
                        </td>
                        <td className="mono muted">{record.rollout_id}</td>
                        <td>
                          <span className={`role role--${record.role}`}>{record.role}</span>
                        </td>
                        <td className="record-content" title={record.content}>
                          {preview(record.content)}
                        </td>
                        <td className="num mono">{signal == null ? "—" : signal.toFixed(3)}</td>
                        <td className="mono muted">{record.policy_version ?? "—"}</td>
                        <td className="mono muted">{fmtTime(Date.parse(record.created_at))}</td>
                        <td>
                          {record.payload_size != null ? (
                            <a
                              className="iconbtn iconbtn--inline"
                              href={experimentBlobUrl(name, record.id)}
                              download
                              title={`Download ${fmtBytes(record.payload_size)}`}
                              aria-label="download blob"
                            >
                              <DownloadIcon />
                            </a>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>
                      </tr>
                      {open && (
                        <tr className="record-detail-row">
                          <td colSpan={9}>
                            <RecordDetails record={record} name={name} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
            </tbody>
          </table>
        )}
        {!records.isLoading && !records.isError && rows.length === 0 && (
          <div className="empty">No rollout records match these filters.</div>
        )}
      </div>

      {(page > 0 || hasMore) && (
        <div className="pager records-pager">
          <span className="pager__info">page {page + 1}</span>
          <button className="btn btn--ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>
            ← Prev
          </button>
          <button
            className="btn btn--ghost"
            disabled={!hasMore}
            onClick={() => setPage(page + 1)}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}

function Drawer({
  name,
  tab,
  expandedId,
}: {
  name: string;
  tab: "records" | "overview";
  expandedId: string | null;
}) {
  const navigate = useNavigate();
  const encName = encodeURIComponent(name);
  const close = () => navigate("/experiments");
  const detail = useQuery({
    queryKey: ["experiment", name],
    queryFn: () => getExperiment(name),
  });
  const d = detail.data;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && navigate("/experiments");
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate]);

  // Toggling a record maps to a URL: expanded → /records/:id, collapsed → base.
  const onToggle = (id: string | null) => {
    if (id) navigate(`/experiments/${encName}/records/${encodeURIComponent(id)}`);
    else navigate(`/experiments/${encName}`);
  };

  return (
    <>
      <div className="scrim" onClick={close} />
      <aside className="drawer">
        <div className="drawer__head">
          <div>
            <div className="drawer__title">
              <span className="glyph" />
              {name}
              <button
                className="iconbtn iconbtn--inline drawer__share"
                title="Copy link to this experiment"
                aria-label="copy link"
                onClick={() => copyLink(`/experiments/${encName}`)}
              >
                <LinkIcon />
              </button>
            </div>
            {d && <div className="drawer__uri">{d.uri}</div>}
          </div>
          <button className="iconbtn" onClick={close} aria-label="close">
            <CloseIcon />
          </button>
        </div>

        <div className="drawer__tabs" role="tablist">
          <button
            className={tab === "records" ? "drawer__tab drawer__tab--active" : "drawer__tab"}
            onClick={() => navigate(`/experiments/${encName}`)}
            role="tab"
            aria-selected={tab === "records"}
          >
            Records
          </button>
          <button
            className={tab === "overview" ? "drawer__tab drawer__tab--active" : "drawer__tab"}
            onClick={() => navigate(`/experiments/${encName}/overview`)}
            role="tab"
            aria-selected={tab === "overview"}
          >
            Overview
          </button>
        </div>

        <div className="drawer__body">
          {tab === "overview" && detail.isLoading && (
            <div className="loading">
              <span className="spinner" />
              loading experiment…
            </div>
          )}
          {tab === "records" && (
            <RecordsView name={name} expandedId={expandedId} onToggle={onToggle} />
          )}
          {tab === "overview" && d && (
            <>
              <div className="section-label">Storage</div>
              <div className="metric-grid">
                <Metric label="Rows" value={fmtInt(d.row_count)} />
                <Metric label="Fragments" value={fmtInt(d.fragment_count)} />
                <Metric label="Pending WAL gens" value={String(d.pending_wal_generations)} />
                <Metric label="Total compactions" value={String(d.total_compactions)} />
                <Metric label="Last updated" value={fmtTime(d.last_updated)} />
                <Metric label="Last compaction" value={fmtTime(d.last_compaction)} />
                <Metric label="Last scan" value={fmtTime(d.scanned_at)} wide />
              </div>
            </>
          )}
        </div>

        {tab === "overview" && (
          <div className="drawer__actions">
            <CompactButton name={name} variant="accent" />
            <MergeWalButton name={name} />
            <IndexIdButton name={name} />
          </div>
        )}
      </aside>
    </>
  );
}

/* ---- stat strip ---------------------------------------------------------- */

function StatCard({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="stat">
      <div className="stat__label">{label}</div>
      <div className="stat__value">
        {value}
        {unit && <span className="unit">{unit}</span>}
      </div>
    </div>
  );
}

/* ---- task queue view ----------------------------------------------------- */

/** Generic state pill for a scheduler task (queued/running/done/failed). */
function TaskStatePill({ state, title }: { state: TaskState; title?: string }) {
  const cls =
    state === "queued"
      ? "pill--queued"
      : state === "running"
        ? "pill--running"
        : state === "done"
          ? "pill--done"
          : "pill--failed";
  return (
    <span className={`pill ${cls}`} title={title}>
      <span className="pill__dot" />
      {state}
    </span>
  );
}

function taskDuration(t: TaskRecord): string {
  const end = t.finished_at ?? Date.now();
  const start = t.started_at ?? t.enqueued_at;
  if (!start) return "—";
  const s = Math.max(0, Math.round((end - start) / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

/** "Merge WAL" trigger — enqueues a merge_wal task for one experiment. */
function MergeWalButton({ name }: { name: string }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const trigger = useMutation({
    mutationFn: () => enqueueTask({ kind: "merge_wal", target: name }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tasks"] });
      navigate("/tasks");
    },
  });
  return (
    <button
      className="btn btn--ghost"
      onClick={() => trigger.mutate()}
      disabled={trigger.isPending}
    >
      {trigger.isPending ? "Merging…" : "Merge WAL"}
    </button>
  );
}

/** "Index id" trigger — enqueues an index_id task (build ZoneMap on id). */
function IndexIdButton({ name }: { name: string }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const trigger = useMutation({
    mutationFn: () => enqueueTask({ kind: "index_id", target: name }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tasks"] });
      navigate("/tasks");
    },
  });
  return (
    <button
      className="btn btn--ghost"
      onClick={() => trigger.mutate()}
      disabled={trigger.isPending}
    >
      {trigger.isPending ? "Indexing…" : "Index id"}
    </button>
  );
}

function TaskQueue() {
  const tasks = useQuery({
    queryKey: ["tasks"],
    queryFn: () => listTasks(),
    refetchInterval: 1000,
  });
  const rows = tasks.data?.tasks ?? [];
  const active = rows.filter((t) => t.state === "queued" || t.state === "running").length;

  return (
    <>
      <div className="stats">
        <StatCard label="Tasks" value={fmtInt(rows.length)} />
        <StatCard label="Active" value={fmtInt(active)} />
      </div>
      <div className="table-wrap">
        {tasks.isError && <div className="error">{String(tasks.error)}</div>}
        {!tasks.isError && (
          <table className="grid">
            <thead>
              <tr>
                <th>Kind</th>
                <th>Target</th>
                <th>State</th>
                <th>Detail</th>
                <th className="num">Duration</th>
                <th>Enqueued</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((t) => (
                <tr key={t.id}>
                  <td>{t.kind}</td>
                  <td>{t.target}</td>
                  <td>
                    <TaskStatePill state={t.state} title={t.error ?? undefined} />
                  </td>
                  <td className="muted">{t.detail ?? t.error ?? "—"}</td>
                  <td className="num">{taskDuration(t)}</td>
                  <td className="muted">{relTime(t.enqueued_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!tasks.isLoading && !tasks.isError && rows.length === 0 && (
          <div className="empty">No tasks in the queue.</div>
        )}
      </div>
    </>
  );
}

/* ---- app ----------------------------------------------------------------- */

const SQL_PLACEHOLDER =
  "SELECT problem_id, count(*) AS n\nFROM records\nGROUP BY problem_id\nORDER BY n DESC";

/** Render one SQL result cell: objects/arrays as compact JSON, null as em dash. */
function sqlCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** Read-only SQL console: pick an experiment, run a SELECT against `records`. */
function SqlConsole() {
  const experiments = useQuery({
    queryKey: ["experiments", "sql-picker"],
    queryFn: () => listExperiments("", 1000, 0),
  });
  const options = experiments.data?.experiments ?? [];

  const [selected, setSelected] = useState("");
  const [sql, setSql] = useState(SQL_PLACEHOLDER);

  // Default the picker to the first experiment once the list loads.
  useEffect(() => {
    if (!selected && options.length > 0) setSelected(options[0].name);
  }, [selected, options]);

  const run = useMutation<SqlQueryResponse, Error>({
    mutationFn: () => runSql(selected, sql),
  });

  const canRun = Boolean(selected) && sql.trim().length > 0 && !run.isPending;
  const submit = () => {
    if (canRun) run.mutate();
  };
  const onKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      submit();
    }
  };

  const result = run.data;

  return (
    <>
      <div className="sql-bar">
        <select
          className="sql-select"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={experiments.isLoading}
        >
          {options.length === 0 && <option value="">No experiments</option>}
          {options.map((exp) => (
            <option key={exp.name} value={exp.name}>
              {exp.name}
            </option>
          ))}
        </select>
        <button className="btn btn--accent" onClick={submit} disabled={!canRun}>
          {run.isPending ? "Running…" : "Run ⌘⏎"}
        </button>
      </div>
      <textarea
        className="sql-editor"
        value={sql}
        spellCheck={false}
        placeholder={SQL_PLACEHOLDER}
        onChange={(e) => setSql(e.target.value)}
        onKeyDown={onKeyDown}
      />
      <p className="sql-hint muted">
        Read-only. Records are exposed as a table named{" "}
        <code>records</code> (merged base + pending WAL view).
      </p>

      {run.isError && <div className="error">{String(run.error?.message ?? run.error)}</div>}

      {result && (
        <>
          <div className="sql-meta muted">
            {fmtInt(result.row_count)} row{result.row_count === 1 ? "" : "s"}
            {result.truncated && " (truncated)"}
          </div>
          <div className="table-wrap">
            {result.columns.length === 0 ? (
              <div className="empty">Query returned no columns.</div>
            ) : (
              <table className="grid">
                <thead>
                  <tr>
                    {result.columns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, ri) => (
                    <tr key={ri}>
                      {row.map((cell, ci) => (
                        <td key={ci} className="muted">
                          {sqlCell(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {result.rows.length === 0 && result.columns.length > 0 && (
              <div className="empty">No rows.</div>
            )}
          </div>
        </>
      )}
    </>
  );
}

/** The experiments list view (stat strip + search + table + pager). The
 * experiment drawer is rendered by the route, layered on top of this list. */
function ExperimentsList() {
  const { search, page, pageSize, setSearch, setPage } = useUiStore();
  const list = useQuery({
    queryKey: ["experiments", search, page, pageSize],
    queryFn: () => listExperiments(search, pageSize, page * pageSize),
    placeholderData: (prev) => prev,
  });

  const total = list.data?.total ?? 0;
  const maxPage = Math.max(0, Math.ceil(total / pageSize) - 1);
  const rows = list.data?.experiments ?? [];

  // Aggregate the visible page for the stat strip.
  const pageRows = rows.reduce((a, e) => a + e.row_count, 0);
  const pageFrags = rows.reduce((a, e) => a + e.fragment_count, 0);
  const hot = rows.filter((e) => e.fragment_count >= 16).length;

  return (
    <>
      <div className="stats">
        <StatCard label="Experiments" value={fmtInt(total)} />
        <StatCard label="Rows (page)" value={fmtCompact(pageRows)} />
        <StatCard label="Fragments (page)" value={fmtCompact(pageFrags)} />
        <StatCard label="Needs compaction" value={String(hot)} unit={hot === 1 ? "exp" : "exps"} />
      </div>

      <div className="toolbar">
        <div className="search">
          <SearchIcon />
          <input
            placeholder="Search by experiment name…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="topbar__spacer" />
        <span className="toolbar__count">{fmtInt(total)} total</span>
      </div>

      <div className="table-wrap">
        {list.isError && <div className="error">{String(list.error)}</div>}
        {!list.isError && (
          <table className="grid">
            <thead>
              <tr>
                <th>Experiment</th>
                <th className="num">Rows</th>
                <th className="num">Fragments</th>
                <th className="num">Pending</th>
                <th>Updated</th>
                <th>Compacted</th>
                <th style={{ textAlign: "right" }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {list.isLoading &&
                Array.from({ length: 6 }).map((_, i) => (
                  <tr key={i}>
                    {Array.from({ length: 7 }).map((__, j) => (
                      <td key={j}>
                        <div className="skeleton" style={{ width: j === 0 ? "60%" : "40%" }} />
                      </td>
                    ))}
                  </tr>
                ))}
              {!list.isLoading && rows.map((exp) => <Row key={exp.name} exp={exp} />)}
            </tbody>
          </table>
        )}
        {!list.isLoading && !list.isError && rows.length === 0 && (
          <div className="empty">
            No experiments{search ? ` matching “${search}”` : ""}.
          </div>
        )}
      </div>

      {maxPage > 0 && (
        <div className="pager">
          <span className="pager__info">
            page {page + 1} / {maxPage + 1}
          </span>
          <button className="btn btn--ghost" disabled={page <= 0} onClick={() => setPage(page - 1)}>
            ← Prev
          </button>
          <button
            className="btn btn--ghost"
            disabled={page >= maxPage}
            onClick={() => setPage(page + 1)}
          >
            Next →
          </button>
        </div>
      )}
    </>
  );
}

/** Experiments list plus, when the URL names an experiment, its detail drawer.
 * `tab` and `recordId` come from the matched route so the drawer state (which
 * tab, which expanded record) is fully reflected in — and driven by — the URL. */
function ExperimentsRoute({ tab }: { tab: "records" | "overview" }) {
  const params = useParams();
  const name = params.name ? decodeURIComponent(params.name) : null;
  const recordId = params.recordId ? decodeURIComponent(params.recordId) : null;
  return (
    <>
      <ExperimentsList />
      {name && (
        <Drawer key={name} name={name} tab={tab} expandedId={recordId} />
      )}
    </>
  );
}

/** Top nav shared across all routes. */
function TopBar() {
  const fetching = useIsFetching();
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand__mark" />
        <span className="brand__name">lance-context</span>
        <span className="brand__sub">control plane</span>
      </div>
      <div className="topbar__spacer" />
      <nav className="tabs">
        <NavLink
          to="/experiments"
          className={({ isActive }) => `tab ${isActive ? "tab--active" : ""}`}
        >
          Experiments
        </NavLink>
        <NavLink to="/tasks" className={({ isActive }) => `tab ${isActive ? "tab--active" : ""}`}>
          Task Queue
        </NavLink>
        <NavLink to="/sql" className={({ isActive }) => `tab ${isActive ? "tab--active" : ""}`}>
          SQL
        </NavLink>
      </nav>
      <div className="live">
        <span className="live__dot" />
        {fetching > 0 ? "syncing" : "live"}
      </div>
    </header>
  );
}

export function App() {
  return (
    <div className="app">
      <TopBar />
      <Routes>
        <Route path="/" element={<Navigate to="/experiments" replace />} />
        <Route path="/experiments" element={<ExperimentsRoute tab="records" />} />
        <Route
          path="/experiments/:name"
          element={<ExperimentsRoute tab="records" />}
        />
        <Route
          path="/experiments/:name/overview"
          element={<ExperimentsRoute tab="overview" />}
        />
        <Route
          path="/experiments/:name/records/:recordId"
          element={<ExperimentsRoute tab="records" />}
        />
        <Route path="/tasks" element={<TaskQueue />} />
        <Route path="/sql" element={<SqlConsole />} />
        <Route path="*" element={<Navigate to="/experiments" replace />} />
      </Routes>
    </div>
  );
}
