//! A protocol activation survives configuration rollback. This records the
//! required protocol, not proof that pre-upgrade writers have drained.
use crate::{Coordinator, Result};
use etcd_client::{Compare, CompareOp, TxnOp};

impl Coordinator {
    pub async fn legacy_fencing_mode(&self, enable: bool) -> Result<bool> {
        let key = format!(
            "{}/worker-merge-admission-mode",
            self.prefix.trim_end_matches('/')
        );
        self.persist_fencing_mode(&key, enable).await
    }

    /// Per-writer activation permits a rolling transition without changing the
    /// admission mode of other workers or newly started masters. The existing
    /// cluster marker remains authoritative and is never downgraded.
    pub async fn worker_fencing_mode(&self, writer_id: &str, enable: bool) -> Result<bool> {
        if writer_id.is_empty() || writer_id.len() > 512 {
            return Err("stable writer identity required for worker fencing".into());
        }
        let cluster_mode = self.legacy_fencing_mode(false).await?;
        let encoded: String = writer_id
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let key = format!(
            "{}/worker-merge-admission-modes/{encoded}",
            self.prefix.trim_end_matches('/')
        );
        let worker_mode = self.persist_fencing_mode(&key, enable).await?;
        Ok(cluster_mode || worker_mode)
    }

    async fn persist_fencing_mode(&self, key: &str, enable: bool) -> Result<bool> {
        let mut client = self.client.clone();
        if enable {
            // No lease and no API to reopen legacy admission. Do not overwrite
            // a future protocol marker during a binary/config rollback.
            self.transact(
                vec![Compare::version(key, CompareOp::Equal, 0)],
                vec![TxnOp::put(key, "fenced-v1", None)],
            )
            .await?;
        }
        let reply = client.get(key, None).await.map_err(|e| e.to_string())?;
        match reply.kvs().first().map(|kv| kv.value()) {
            None if !enable => Ok(false),
            None => Err("worker merge protocol activation was not persisted".into()),
            Some(b"fenced-v1") => Ok(true),
            Some(_) => {
                Err("unknown worker merge admission protocol; refusing legacy fallback".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn legacy_retirement_worker_activation_is_isolated_and_survives_config_rollback() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let prefix = format!("/merge-mode/{}", crate::Execution::new("", "", "", 1).id);
        let first = Coordinator::new(
            etcd_client::Client::connect([&endpoint], None)
                .await
                .unwrap(),
            prefix.clone(),
        );
        assert!(first.worker_fencing_mode("worker-9", true).await.unwrap());
        assert!(!first.legacy_fencing_mode(false).await.unwrap());
        assert!(!first.worker_fencing_mode("worker-8", false).await.unwrap());
        let restarted = Coordinator::new(
            etcd_client::Client::connect([&endpoint], None)
                .await
                .unwrap(),
            prefix,
        );
        assert!(restarted
            .worker_fencing_mode("worker-9", false)
            .await
            .unwrap());
        assert!(!restarted
            .worker_fencing_mode("worker-8", false)
            .await
            .unwrap());
        assert!(first.legacy_fencing_mode(true).await.unwrap());
        assert!(restarted
            .worker_fencing_mode("worker-8", false)
            .await
            .unwrap());
        assert!(restarted.worker_fencing_mode("", true).await.is_err());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn legacy_retirement_unknown_worker_protocol_cannot_be_overwritten() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let prefix = format!("/merge-mode/{}", crate::Execution::new("", "", "", 1).id);
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        client
            .put(
                format!("{prefix}/worker-merge-admission-modes/61"),
                "fenced-v2",
                None,
            )
            .await
            .unwrap();
        let coordinator = Coordinator::new(client, prefix);
        assert!(coordinator.worker_fencing_mode("a", false).await.is_err());
        assert!(coordinator.worker_fencing_mode("a", true).await.is_err());
        assert!(coordinator.legacy_fencing_mode(true).await.unwrap());
        assert!(coordinator.worker_fencing_mode("a", false).await.is_err());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn activated_protocol_survives_a_new_client_and_disabled_flag() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let prefix = format!("/merge-mode/{}", crate::Execution::new("", "", "", 1).id);
        let first = Coordinator::new(
            etcd_client::Client::connect([&endpoint], None)
                .await
                .unwrap(),
            prefix.clone(),
        );
        assert!(!first.legacy_fencing_mode(false).await.unwrap());
        assert!(first.legacy_fencing_mode(true).await.unwrap());
        let restarted = Coordinator::new(
            etcd_client::Client::connect([&endpoint], None)
                .await
                .unwrap(),
            prefix,
        );
        assert!(restarted.legacy_fencing_mode(false).await.unwrap());
    }
    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn a_newer_protocol_cannot_be_replaced_by_activation() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let prefix = format!("/merge-mode/{}", crate::Execution::new("", "", "", 1).id);
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        client
            .put(
                format!("{prefix}/worker-merge-admission-mode"),
                "fenced-v2",
                None,
            )
            .await
            .unwrap();
        let c = Coordinator::new(client, prefix);
        assert!(c.legacy_fencing_mode(false).await.is_err());
        assert!(c.legacy_fencing_mode(true).await.is_err());
    }
}
