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
        let mut client = self.client.clone();
        if enable {
            // No lease and no API to reopen legacy admission. Do not overwrite
            // a future protocol marker during a binary/config rollback.
            self.transact(
                vec![Compare::version(key.as_str(), CompareOp::Equal, 0)],
                vec![TxnOp::put(key.as_str(), "fenced-v1", None)],
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
