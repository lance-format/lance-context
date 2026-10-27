//! Observe a preparation's two ownership keys without periodic claim polling.
//! This never grants write permission; promotion still performs its own CAS.
use super::{etcd_error, TaskClaim, TaskStore};
use etcd_client::{EventType, Txn, TxnOp, TxnOpResponse, WatchOptions, WatchStream};
use lance_context_metrics::etcd::{ObserveEtcd, Operation};
use std::time::Duration;
use tokio::time::Instant;

const REFRESH: Duration = Duration::from_secs(30);

pub(crate) struct PreparationObserver {
    client: etcd_client::Client,
    keys: [String; 2],
    token: Vec<u8>,
    stream: Option<WatchStream>,
    revision: i64,
    refresh: Instant,
}

impl TaskStore {
    pub(crate) fn preparation_observer(
        &self,
        claim: &TaskClaim,
    ) -> lance::Result<PreparationObserver> {
        let preparation = claim
            .backend
            .preparation_key
            .clone()
            .ok_or_else(|| lance::Error::io("maintenance preparation key missing"))?;
        Ok(PreparationObserver {
            client: self.inner.client.clone(),
            keys: [claim.backend.claim_key.clone(), preparation],
            token: claim.backend.token.as_bytes().to_vec(),
            stream: None,
            revision: 0,
            refresh: Instant::now(),
        })
    }
}

impl PreparationObserver {
    pub(crate) async fn observe(&mut self) -> lance::Result<bool> {
        let result = self.observe_inner().await;
        if result.is_err() {
            // A broken/compacted stream is uncertainty. The caller paces one
            // retry, which captures a fresh revision before subscribing again.
            self.stream = None;
        }
        result
    }

    async fn capture(&mut self) -> lance::Result<bool> {
        metrics::counter!("master_preparation_owner_snapshots_total").increment(1);
        let response = self
            .client
            .clone()
            .txn(
                Txn::new().and_then(
                    self.keys
                        .iter()
                        .map(|key| TxnOp::get(key.as_str(), None))
                        .collect::<Vec<_>>(),
                ),
            )
            .observed(Operation::TaskReadTxn)
            .await
            .map_err(etcd_error("read preparation ownership"))?;
        let revision = response
            .header()
            .map(|h| h.revision())
            .filter(|r| *r > 0 && *r < i64::MAX)
            .ok_or_else(|| lance::Error::io("preparation snapshot revision missing"))?;
        let values = response.op_responses();
        if values.len() != self.keys.len() {
            return Err(lance::Error::io("preparation snapshot incomplete"));
        }
        for value in values {
            let TxnOpResponse::Get(value) = value else {
                return Err(lance::Error::io("invalid preparation snapshot"));
            };
            if value.kvs().first().map(|kv| kv.value()) != Some(self.token.as_slice()) {
                return Ok(false);
            }
        }
        self.revision = revision;
        self.refresh = Instant::now() + REFRESH;
        Ok(true)
    }

    async fn subscribe(&mut self) -> lance::Result<()> {
        let options = WatchOptions::new().with_start_revision(self.revision + 1);
        let mut stream = self
            .client
            .clone()
            .watch(self.keys[0].clone(), Some(options.clone()))
            .observed(Operation::DispatchWatch)
            .await
            .map_err(etcd_error("watch preparation claim"))?;
        stream
            .watch(self.keys[1].clone(), Some(options))
            .await
            .map_err(etcd_error("watch preparation key"))?;
        self.stream = Some(stream);
        metrics::counter!("master_preparation_owner_watches_total").increment(1);
        Ok(())
    }

    async fn observe_inner(&mut self) -> lance::Result<bool> {
        loop {
            if self.stream.is_none() || Instant::now() >= self.refresh {
                if !self.capture().await? {
                    return Ok(false);
                }
                if self.stream.is_none() {
                    self.subscribe().await?;
                }
                return Ok(true);
            }
            let response = match tokio::time::timeout_at(
                self.refresh,
                self.stream
                    .as_mut()
                    .expect("ownership stream opened")
                    .message(),
            )
            .await
            {
                Err(_) => continue,
                Ok(result) => result.map_err(etcd_error("observe preparation ownership"))?,
            };
            let response =
                response.ok_or_else(|| lance::Error::io("preparation ownership watch ended"))?;
            if response.canceled() {
                return Err(lance::Error::io("preparation ownership watch canceled"));
            }
            for event in response.events() {
                let kv = event
                    .kv()
                    .ok_or_else(|| lance::Error::io("preparation event missing key"))?;
                if kv.mod_revision() <= self.revision {
                    continue;
                }
                if self.keys.iter().any(|key| key.as_bytes() == kv.key())
                    && (event.event_type() == EventType::Delete || kv.value() != self.token)
                {
                    return Ok(false);
                }
            }
            // Creation acknowledgements/unchanged values do not reset the
            // refresh deadline or the caller's actual-work idle budget.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn observer() -> PreparationObserver {
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let mut client =
            etcd_client::Client::connect(endpoints.split(',').collect::<Vec<_>>(), None)
                .await
                .unwrap();
        let prefix = format!(
            "/preparation-watch-test/{}",
            lance_context_core::generate_id()
        );
        let keys = [format!("{prefix}/claim"), format!("{prefix}/preparation")];
        for key in &keys {
            client.put(key.as_str(), "owner", None).await.unwrap();
        }
        PreparationObserver {
            client,
            keys,
            token: b"owner".to_vec(),
            stream: None,
            revision: 0,
            refresh: Instant::now(),
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn preparation_watch_is_quiet_and_replays_loss_without_another_snapshot() {
        for key_index in 0..2 {
            let mut observer = observer().await;
            assert!(observer.observe().await.unwrap());
            let revision = observer.revision;
            let mut client = observer.client.clone();
            // Unrelated writes advance etcd's revision. A hidden periodic
            // snapshot would advance the observer's revision too.
            client
                .put("/preparation-watch-unrelated", "changed", None)
                .await
                .unwrap();
            assert!(
                tokio::time::timeout(Duration::from_secs(5), observer.observe())
                    .await
                    .is_err()
            );
            assert_eq!(observer.revision, revision);
            client
                .delete(observer.keys[key_index].clone(), None)
                .await
                .unwrap();
            assert!(
                !tokio::time::timeout(Duration::from_secs(2), observer.observe())
                    .await
                    .unwrap()
                    .unwrap()
            );
            assert_eq!(observer.revision, revision, "loss consumed from Watch");
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn preparation_loss_between_snapshot_and_subscription_is_replayed() {
        for key_index in 0..2 {
            let mut observer = observer().await;
            assert!(observer.capture().await.unwrap());
            let revision = observer.revision;
            observer
                .client
                .clone()
                .delete(observer.keys[key_index].clone(), None)
                .await
                .unwrap();
            observer.subscribe().await.unwrap();
            assert!(
                !tokio::time::timeout(Duration::from_secs(2), observer.observe())
                    .await
                    .unwrap()
                    .unwrap()
            );
            assert_eq!(observer.revision, revision);
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn preparation_refresh_reuses_stream_and_reconnect_rechecks_ownership() {
        let mut observer = observer().await;
        assert!(observer.observe().await.unwrap());
        let revision = observer.revision;
        let mut client = observer.client.clone();
        client
            .put("/preparation-watch-unrelated", "next", None)
            .await
            .unwrap();
        observer.refresh = Instant::now();
        assert!(observer.observe().await.unwrap());
        assert!(observer.revision > revision);
        // A disconnected observer must re-read, not trust its old token.
        observer.stream = None;
        client
            .put(observer.keys[0].clone(), "replacement", None)
            .await
            .unwrap();
        assert!(!observer.observe().await.unwrap());
        assert!(observer.stream.is_none());
    }
}
