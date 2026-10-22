//! Recoverable HTTP producers sharing one bounded group-commit journal and one
//! HTTP connection pool. Persist an intent before sending, and persist its ACK
//! before releasing the body or advancing the sequence. Local journal ambiguity
//! fences that lane until reopen; network ambiguity retains the exact intent.
use super::*;
use lance_context_wal::client_journal::{Journal, Settings};
use std::{collections::BTreeSet, path::PathBuf};
use tokio::sync::Mutex as AsyncMutex;

const MAX_METADATA: usize = 32 << 10;
const MAX_TOKEN: usize = 12 << 10;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pending {
    sequence: u64,
    bytes: usize,
    digest: [u8; 32],
    #[serde(skip)]
    body: Bytes,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    version: u8,
    next: u64,
    producer: Option<Opened>,
    last: Option<Accepted>,
    pending: Option<Pending>,
}
impl Default for Saved {
    fn default() -> Self {
        Self {
            version: 1,
            next: 1,
            producer: None,
            last: None,
            pending: None,
        }
    }
}
struct Lane {
    saved: Saved,
    reopen: bool,
}

pub struct Config {
    pub directory: PathBuf,
    pub opens: Vec<ProducerOpen>,
    pub base: reqwest::Url,
    pub request_timeout: Duration,
    pub journal: Settings,
}

pub struct DurableClient {
    // Drop/join the journal before releasing the logical workspace reservation.
    journal: Journal,
    lanes: Vec<AsyncMutex<Lane>>,
    opens: Vec<ProducerOpen>,
    groups: Vec<u64>,
    scope: Scope,
    scope_hash: [u8; 32],
    client: reqwest::Client,
    base: reqwest::Url,
    timeout: Duration,
    _workspace: Arc<MergeReservation>,
}
fn digest(body: &[u8]) -> [u8; 32] {
    ring::digest::digest(&ring::digest::SHA256, body)
        .as_ref()
        .try_into()
        .unwrap()
}
impl DurableClient {
    /// Reopen with the same layout, binding, ordered producer-open tuples and
    /// journal settings. Persist those tuples in configuration; generating new
    /// operation UUIDs on restart is not a retry. Endpoint/pool may change for
    /// gateway failover. Supply a trusted HTTP client with redirects disabled.
    pub async fn open(
        config: Config,
        layout: &Layout,
        binding: TableBinding,
        client: reqwest::Client,
        budget: Arc<MergeMemoryBudget>,
    ) -> Result<Self> {
        let Config {
            directory,
            opens,
            mut base,
            request_timeout: timeout,
            journal: settings,
        } = config;
        settings.validate()?;
        layout.validate()?;
        let contract = binding.contract()?;
        ensure!(
            !binding.table.is_nil()
                && !binding.incarnation.is_nil()
                && (1..=16384).contains(&opens.len())
                && (Duration::from_millis(100)..=Duration::from_secs(120)).contains(&timeout)
                && matches!(base.scheme(), "http" | "https")
                && base.query().is_none()
                && base.fragment().is_none(),
            "invalid HTTP client scope"
        );
        let scope = Scope {
            cluster: layout.cluster,
            layout: layout.fingerprint()?,
            table: binding.table,
            destination: binding.destination,
            incarnation: binding.incarnation,
            contract,
        };
        let scope_hash = super::super::receipt::scope_hash(&scope)?;
        let mut used = BTreeSet::new();
        let groups = opens
            .iter()
            .map(|open| {
                let group = layout.group_for_bucket(open.bucket)?;
                ensure!(
                    !open.operation.is_nil()
                        && open.expected_generation < u64::MAX
                        && open.slot < producer::MAX_SLOTS
                        && used.insert((group, open.slot)),
                    "invalid/duplicate producer slot"
                );
                Ok(group)
            })
            .collect::<Result<Vec<_>>>()?;
        let per_lane = scope.contract.max_record_bytes as usize + MAX_METADATA + 4;
        ensure!(
            per_lane
                .checked_mul(opens.len())
                .is_some_and(|n| n <= settings.state_bytes as usize)
                && per_lane + 128 <= settings.batch_bytes as usize,
            "journal capacity cannot hold the configured producer intents"
        );
        let workspace = (settings.state_bytes as usize)
            .checked_mul(3)
            .and_then(|n| n.checked_add(settings.queue_bytes as usize))
            .and_then(|n| n.checked_add(opens.len() * MAX_METADATA * 2))
            .context("journal workspace overflow")?;
        ensure!(
            workspace <= budget.limit(),
            "journal workspace exceeds client budget"
        );
        // Hold before recovery allocates state or the disk worker can encode it.
        let reservation = Arc::new(budget.reserve(workspace).await);
        base.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("HTTP base URL cannot contain path segments"))?
            .pop_if_empty()
            .extend(["api", "v1", "ingest", &scope.contract.qualified_table]);
        tokio::task::spawn_blocking(move || {
            let workspace_guard = reservation;
            let binding_digest = digest(&serde_json::to_vec(&(&scope, &opens))?);
            let count = opens.len() as u32;
            let (journal, recovered) = Journal::open(&directory, &binding_digest, count, settings)?;
            let mut lanes = Vec::with_capacity(opens.len());
            for (key, value) in recovered.into_iter().enumerate() {
                let saved = if let Some(bytes) = value {
                    ensure!(bytes.len() >= 4, "truncated client journal record");
                    let length = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
                    ensure!(
                        length > 0 && length <= MAX_METADATA && length <= bytes.len() - 4,
                        "invalid client journal metadata"
                    );
                    let mut saved: Saved = serde_json::from_slice(&bytes[4..4 + length])?;
                    let body = bytes.slice(4 + length..);
                    if let Some(pending) = saved.pending.as_mut() {
                        ensure!(
                            pending.bytes == body.len()
                                && !body.is_empty()
                                && body.len() <= scope.contract.max_record_bytes as usize
                                && pending.digest == digest(&body)
                                && pending.sequence == saved.next,
                            "invalid recovered append intent"
                        );
                        pending.body = body;
                    } else {
                        ensure!(body.is_empty(), "unexpected journal payload");
                    }
                    saved
                } else {
                    Saved::default()
                };
                ensure!(
                    saved.version == 1
                        && saved.next > 0
                        && saved.next < u64::MAX
                        && (saved.last.is_none() || saved.producer.is_some()),
                    "invalid recovered sequence"
                );
                ensure!(
                    saved.last.as_ref().map_or(saved.next == 1, |last| {
                        last.sequence.checked_add(1) == Some(saved.next)
                            && last.group == groups[key]
                            && last.index > 0
                            && last.index < u64::MAX
                            && last.rows > 0
                            && last.rows <= scope.contract.max_rows
                            && !last.receipt.is_empty()
                            && last.receipt.len() <= MAX_TOKEN
                    }),
                    "invalid recovered ACK"
                );
                if let Some(credential) = &saved.producer {
                    ensure!(
                        credential.scope == scope_hash
                            && credential.generation == opens[key].expected_generation + 1
                            && !credential.producer.is_empty()
                            && credential.producer.len() <= MAX_TOKEN,
                        "invalid recovered producer"
                    );
                }
                lanes.push(AsyncMutex::new(Lane {
                    saved,
                    reopen: false,
                }));
            }
            Ok(Self {
                journal,
                lanes,
                opens,
                groups,
                scope,
                scope_hash,
                client,
                base,
                timeout,
                _workspace: workspace_guard,
            })
        })
        .await?
    }
    fn lane(&self, key: usize) -> Result<&AsyncMutex<Lane>> {
        self.lanes.get(key).context("client lane out of range")
    }
    fn endpoint(&self, action: &str) -> reqwest::Url {
        let mut url = self.base.clone();
        url.path_segments_mut().unwrap().push(action);
        url
    }
    async fn persist(&self, key: usize, lane: &mut Lane, next: Saved) -> Result<()> {
        ensure!(
            !lane.reopen,
            "local journal outcome unknown; reopen before continuing"
        );
        let metadata = serde_json::to_vec(&next)?;
        ensure!(
            metadata.len() <= MAX_METADATA,
            "client journal metadata too large"
        );
        let mut prefix = Vec::with_capacity(4 + metadata.len());
        prefix.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        prefix.extend_from_slice(&metadata);
        let body = next
            .pending
            .as_ref()
            .map_or_else(Bytes::new, |p| p.body.clone());
        // Cancellation at this await can already have changed disk. Keep the
        // fence unless a durable completion and in-memory update both finish.
        lane.reopen = true;
        self.journal
            .put(key as u32, Bytes::from(prefix), body)
            .await?;
        lane.saved = next;
        lane.reopen = false;
        Ok(())
    }
    async fn stage_locked(&self, key: usize, lane: &mut Lane, body: Bytes) -> Result<u64> {
        ensure!(
            !lane.reopen,
            "local journal outcome unknown; reopen before continuing"
        );
        ensure!(
            !body.is_empty() && body.len() <= self.scope.contract.max_record_bytes as usize,
            "append body outside table bounds"
        );
        ensure!(
            lane.saved.next < u64::MAX - 1,
            "producer sequence exhausted"
        );
        if let Some(pending) = &lane.saved.pending {
            let same = if pending.body.len() != body.len() {
                false
            } else if pending.body.as_ptr() == body.as_ptr() {
                true
            } else {
                let original = pending.body.clone();
                let workspace = self._workspace.clone();
                tokio::task::spawn_blocking(move || {
                    let _workspace = workspace;
                    original == body
                })
                .await?
            };
            ensure!(
                same,
                "different payload supplied for an unresolved sequence"
            );
            return Ok(pending.sequence);
        }
        let workspace = self._workspace.clone();
        let body = tokio::task::spawn_blocking(move || {
            let _workspace = workspace;
            (digest(&body), body)
        })
        .await?;
        let mut next = lane.saved.clone();
        let sequence = next.next;
        next.pending = Some(Pending {
            sequence,
            bytes: body.1.len(),
            digest: body.0,
            body: body.1,
        });
        self.persist(key, lane, next).await?;
        Ok(sequence)
    }
    /// Persist a complete source intent without sending it yet. Useful when
    /// coordinating a durable source checkpoint with later recovery.
    pub async fn stage(&self, key: usize, body: Bytes) -> Result<u64> {
        let mut lane = self.lane(key)?.lock().await;
        self.stage_locked(key, &mut lane, body).await
    }
    pub async fn append(&self, key: usize, body: Bytes) -> Result<Accepted> {
        let mut lane = self.lane(key)?.lock().await;
        self.stage_locked(key, &mut lane, body).await?;
        self.resume_locked(key, &mut lane)
            .await?
            .context("staged append disappeared")
    }
    /// Resend only the recovered/staged bytes and sequence. A network error
    /// leaves this intent intact; the caller decides when to retry it.
    pub async fn resume(&self, key: usize) -> Result<Option<Accepted>> {
        let mut lane = self.lane(key)?.lock().await;
        self.resume_locked(key, &mut lane).await
    }
    async fn resume_locked(&self, key: usize, lane: &mut Lane) -> Result<Option<Accepted>> {
        ensure!(
            !lane.reopen,
            "local journal outcome unknown; reopen before continuing"
        );
        let Some(pending) = lane.saved.pending.clone() else {
            return Ok(None);
        };
        if lane.saved.producer.is_none() {
            let response = self
                .client
                .post(self.endpoint("producer"))
                .timeout(self.timeout)
                .json(&self.opens[key])
                .send()
                .await?
                .error_for_status()?;
            let credential: Opened = bounded_json(response).await?;
            ensure!(
                credential.scope == self.scope_hash
                    && credential.generation == self.opens[key].expected_generation + 1
                    && !credential.producer.is_empty()
                    && credential.producer.len() <= MAX_TOKEN,
                "foreign producer credential"
            );
            let mut next = lane.saved.clone();
            next.producer = Some(credential);
            self.persist(key, lane, next).await?;
        }
        let response = self
            .client
            .post(self.endpoint("append"))
            .timeout(self.timeout)
            .header(header::CONTENT_TYPE.as_str(), ARROW_FILE)
            .header(PRODUCER, &lane.saved.producer.as_ref().unwrap().producer)
            .header(SEQUENCE, pending.sequence.to_string())
            .body(pending.body.clone())
            .send()
            .await?;
        ensure!(
            response.status() == reqwest::StatusCode::ACCEPTED,
            "append outcome unresolved (HTTP {}); retain original intent",
            response.status()
        );
        let accepted: Accepted = bounded_json(response).await?;
        ensure!(
            accepted.sequence == pending.sequence
                && accepted.group == self.groups[key]
                && accepted.index > 0
                && accepted.index < u64::MAX
                && accepted.digest == pending.digest
                && accepted.rows > 0
                && accepted.rows <= self.scope.contract.max_rows
                && !accepted.receipt.is_empty()
                && accepted.receipt.len() <= MAX_TOKEN,
            "foreign append ACK"
        );
        let mut next = lane.saved.clone();
        next.next = pending.sequence + 1;
        next.last = Some(accepted.clone());
        next.pending = None;
        self.persist(key, lane, next).await?;
        Ok(Some(accepted))
    }
    pub async fn last_accepted(&self, key: usize) -> Result<Option<Accepted>> {
        let lane = self.lane(key)?.lock().await;
        ensure!(
            !lane.reopen,
            "local journal outcome unknown; reopen before continuing"
        );
        Ok(lane.saved.last.clone())
    }
    pub fn statistics(&self) -> lance_context_wal::client_journal::Statistics {
        self.journal.statistics()
    }
    /// Caller first joins its producer tasks. Move the final owner off the I/O
    /// executor while the local disk worker joins and releases its engine lock.
    pub async fn shutdown(self) -> Result<()> {
        tokio::task::spawn_blocking(move || drop(self)).await?;
        Ok(())
    }
}
async fn bounded_json<T: serde::de::DeserializeOwned>(
    mut response: reqwest::Response,
) -> Result<T> {
    if let Some(length) = response.content_length() {
        ensure!(
            length <= MAX_METADATA as u64,
            "HTTP response metadata too large"
        );
    }
    let mut bytes = Vec::new();
    while let Some(part) = response.chunk().await? {
        ensure!(
            part.len() <= MAX_METADATA - bytes.len(),
            "HTTP response metadata too large"
        );
        bytes.extend_from_slice(&part);
    }
    Ok(serde_json::from_slice(&bytes)?)
}
