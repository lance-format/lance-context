import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  compactionStatus,
  getExperiment,
  listExperiments,
  triggerCompaction,
  type CompactJobStatus,
  type ExperimentSummary,
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
  const select = useUiStore((s) => s.select);
  const hot = exp.fragment_count >= 16;
  return (
    <tr>
      <td>
        <button className="name-cell" onClick={() => select(exp.name)}>
          {exp.name}
        </button>
      </td>
      <td className="num">{fmtInt(exp.row_count)}</td>
      <td className={`num ${hot ? "frag--hot" : ""}`}>{fmtInt(exp.fragment_count)}</td>
      <td className="num">{exp.pending_wal_generations}</td>
      <td className="muted">{relTime(exp.last_updated)}</td>
      <td className="muted">{relTime(exp.last_compaction)}</td>
      <td style={{ textAlign: "right" }}>
        <CompactButton name={exp.name} />
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

function Drawer({ name }: { name: string }) {
  const select = useUiStore((s) => s.select);
  const detail = useQuery({
    queryKey: ["experiment", name],
    queryFn: () => getExperiment(name, true),
  });
  const d = detail.data;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && select(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [select]);

  return (
    <>
      <div className="scrim" onClick={() => select(null)} />
      <aside className="drawer">
        <div className="drawer__head">
          <div>
            <div className="drawer__title">
              <span className="glyph" />
              {name}
            </div>
            {d && <div className="drawer__uri">{d.uri}</div>}
          </div>
          <button className="iconbtn" onClick={() => select(null)} aria-label="close">
            <CloseIcon />
          </button>
        </div>

        <div className="drawer__body">
          {detail.isLoading && (
            <div className="loading">
              <span className="spinner" />
              loading experiment…
            </div>
          )}
          {d && (
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

        <div className="drawer__actions">
          <CompactButton name={name} variant="accent" />
        </div>
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

/* ---- app ----------------------------------------------------------------- */

export function App() {
  const { search, page, pageSize, selected, setSearch, setPage } = useUiStore();
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
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand__mark" />
          <span className="brand__name">lance-context</span>
          <span className="brand__sub">control plane</span>
        </div>
        <div className="topbar__spacer" />
        <div className="live">
          <span className="live__dot" />
          {list.isFetching ? "syncing" : "live"}
        </div>
      </header>

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

      {selected && <Drawer name={selected} />}
    </div>
  );
}
