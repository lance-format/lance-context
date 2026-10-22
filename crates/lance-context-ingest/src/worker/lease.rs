use super::*;
use etcd_client::{Client as Etcd, Compare, CompareOp, GetOptions, PutOptions, Txn, TxnOp};
use std::{future::Future, sync::atomic::AtomicBool};

/// Supply the same authorized etcd prefix to every worker serving these tables.
/// Transport credentials/TLS belong to the application's etcd connection.
#[derive(Clone)]
pub struct Claims {
    client: Etcd,
    prefix: String,
    ttl: i64,
    timeout: Duration,
}
impl Claims {
    pub fn new(client: Etcd, prefix: String, ttl: i64, timeout: Duration) -> Result<Self> {
        ensure!(
            prefix.starts_with('/')
                && !prefix.trim_matches('/').is_empty()
                && prefix.len() <= 1024
                && !prefix.contains('\0')
                && (10..=120).contains(&ttl)
                && (Duration::from_millis(100)..=Duration::from_secs(ttl as u64 / 3))
                    .contains(&timeout),
            "invalid materializer scheduling scope"
        );
        Ok(Self {
            client,
            prefix: prefix.trim_end_matches('/').into(),
            ttl,
            timeout,
        })
    }
    async fn rpc<T>(
        &self,
        operation: impl Future<Output = std::result::Result<T, etcd_client::Error>>,
    ) -> Result<T> {
        Ok(tokio::time::timeout(self.timeout, operation)
            .await
            .context("etcd scheduling deadline")??)
    }
    pub(crate) async fn take(&self, ingress: &Ingress) -> Result<Option<Claim>> {
        let scope = super::super::receipt::scope_hash(&ingress.scope)?;
        let owner = Uuid::new_v4();
        let request = Uuid::new_v4();
        let key = format!(
            "{}/{}/{}",
            self.prefix, ingress.scope.cluster, ingress.scope.table
        );
        // Busy-table probes must not grant/revoke leases on every worker poll:
        // those are replicated writes and can stall unrelated lease renewals.
        // This is only a hint: a stale occupied value delays one attempt, while
        // a stale empty value still must pass the transaction below. The local
        // read avoids a quorum barrier on every occupied-table probe. The CAS
        // and the later claim.valid() check remain linearizable.
        if !self
            .rpc(self.client.clone().get(
                key.clone(),
                Some(GetOptions::new().with_keys_only().with_serializable()),
            ))
            .await
            .context("checking existing materializer claim")?
            .kvs()
            .is_empty()
        {
            return Ok(None);
        }
        let value = serde_json::to_vec(&(scope, owner, request))?;
        let lease = self
            .rpc(self.client.clone().lease_grant(self.ttl, None))
            .await
            .context("granting materializer lease")?
            .id();
        ensure!(lease != 0, "etcd returned an invalid lease");
        let outcome = self
            .rpc(
                self.client.clone().txn(
                    Txn::new()
                        .when([Compare::version(key.clone(), CompareOp::Equal, 0)])
                        .and_then([TxnOp::put(
                            key.clone(),
                            value.clone(),
                            Some(PutOptions::new().with_lease(lease)),
                        )]),
                ),
            )
            .await
            .context("claiming materializer table")?;
        // Unknown outcomes leave only a bounded-TTL lease. Never assume the
        // transaction failed and never delete the key to make acquisition pass.
        if !outcome.succeeded() {
            self.rpc(self.client.clone().lease_revoke(lease))
                .await
                .context("revoking unused materializer lease")?;
            return Ok(None);
        }
        let (mut keeper, mut stream) = self
            .rpc(self.client.clone().lease_keep_alive(lease))
            .await
            .context("starting materializer lease renewal")?;
        let (stop, mut stopped) = watch::channel(false);
        let live = Arc::new(AtomicBool::new(true));
        let state = live.clone();
        let renewal = Duration::from_secs(self.ttl as u64 / 3);
        let timeout = self.timeout;
        let task = tokio::spawn(async move {
            let mut tick = tokio::time::interval(renewal);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = stopped.changed() => break,
                    _ = tick.tick() => {
                        let result = tokio::time::timeout(timeout, async {
                            keeper.keep_alive().await?;
                            let response = stream.message().await?.context("lease renewal stream closed")?;
                            ensure!(response.id() == lease && response.ttl() > 0, "lease expired");
                            Ok::<_, anyhow::Error>(())
                        }).await;
                        if !matches!(result, Ok(Ok(()))) { break; }
                    }
                }
            }
            state.store(false, Ordering::Release);
        });
        Ok(Some(Claim {
            store: self.clone(),
            key,
            value,
            lease,
            scope,
            owner,
            request,
            live,
            stop,
            task: Some(task),
        }))
    }
}
pub(crate) struct Claim {
    store: Claims,
    key: String,
    value: Vec<u8>,
    lease: i64,
    scope: [u8; 32],
    owner: Uuid,
    request: Uuid,
    live: Arc<AtomicBool>,
    stop: watch::Sender<bool>,
    task: Option<JoinHandle<()>>,
}
impl Drop for Claim {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
        // The bounded renewal finishes and the unrevoked lease expires if the
        // owning task panics. No detached delete can remove a successor's claim.
    }
}
impl Claim {
    pub(crate) async fn valid(&self) -> Result<bool> {
        if !self.live.load(Ordering::Acquire) {
            return Ok(false);
        }
        let response = self
            .store
            .rpc(self.store.client.clone().get(self.key.clone(), None))
            .await
            .context("validating materializer claim")?;
        Ok(response
            .kvs()
            .first()
            .is_some_and(|kv| kv.value() == self.value && kv.lease() == self.lease))
    }
    pub(crate) async fn acquire(
        &self,
        assignment: Assignment,
        options: PublishOptions,
        budget: Arc<MergeMemoryBudget>,
    ) -> Result<Option<Publisher>> {
        let ingress = &assignment.ingress;
        ensure!(
            super::super::receipt::scope_hash(&ingress.scope)? == self.scope,
            "claim belongs to another binding"
        );
        // Read the WAL epoch BEFORE the linearizable lease check. A paused old
        // owner must not check its lease first, read the successor epoch later,
        // and use that newer epoch to steal ownership back.
        let query::Value::Publisher(Some(view)) = crate::materializer::query(
            ingress,
            ingress.client.catalog_group(),
            query::Query::Publisher {
                table: ingress.scope.table,
            },
        )
        .await?
        else {
            bail!("publisher binding missing");
        };
        ensure!(
            view.destination == ingress.scope.destination
                && view.ingest == ingress.scope.contract
                && view
                    .manifests
                    .as_ref()
                    .is_some_and(|m| m.incarnation == ingress.scope.incarnation),
            "claim binding changed"
        );
        if !self.valid().await? {
            return Ok(None);
        }
        Ok(Some(
            Publisher::acquire(
                assignment.ingress,
                assignment.storage,
                options,
                budget,
                Acquisition {
                    owner: self.owner,
                    request: self.request,
                    expected_epoch: view.authority.epoch,
                },
            )
            .await?,
        ))
    }
    pub(crate) async fn release(mut self) -> Result<()> {
        let _ = self.stop.send(true);
        self.task
            .take()
            .context("lease keeper already joined")?
            .await?;
        // A lease belongs to exactly this claim. Revoke cannot delete a newer
        // value attached to a successor's different lease, even after expiry.
        self.store
            .rpc(self.store.client.clone().lease_revoke(self.lease))
            .await
            .context("releasing materializer lease")?;
        Ok(())
    }
}
