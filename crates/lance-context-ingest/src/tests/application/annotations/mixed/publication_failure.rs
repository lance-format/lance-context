//! A real filesystem error stops the executable owner before catalog Prepare.
use super::*;
use std::{fs::Permissions, os::unix::fs::PermissionsExt, path::PathBuf};

struct ReadOnlyDirectory {
    path: PathBuf,
    original: Option<Permissions>,
}

impl ReadOnlyDirectory {
    fn new(path: PathBuf) -> Result<Self> {
        let original = std::fs::metadata(&path)?.permissions();
        let guard = Self {
            path,
            original: Some(original.clone()),
        };
        std::fs::set_permissions(
            &guard.path,
            Permissions::from_mode(original.mode() & !0o222),
        )?;
        let probe = guard.path.join("permission-probe");
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
        {
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => Ok(guard),
            Err(error) => Err(error.into()),
            Ok(file) => {
                drop(file);
                std::fs::remove_file(probe)?;
                bail!("publication failure fixture requires an unprivileged process");
            }
        }
    }

    fn restore(mut self) -> Result<()> {
        std::fs::set_permissions(&self.path, self.original.as_ref().unwrap().clone())?;
        self.original = None;
        Ok(())
    }
}

impl Drop for ReadOnlyDirectory {
    fn drop(&mut self) {
        if let Some(original) = &self.original {
            let _ = std::fs::set_permissions(&self.path, original.clone());
        }
    }
}

pub(super) async fn fail(
    c: &network::Cluster,
    root: &Path,
    client: &reqwest::Client,
    ingress: &str,
    producer: &Opened,
    owner: &mut Process,
    acknowledged: u64,
) -> Result<u64> {
    let before = annotation_publication::state(c).await?;
    assert!(before.progress.pending.is_none());
    let blocked = ReadOnlyDirectory::new(root.join("body.lance/data"))?;
    // The ACKed field update is already in the one-record suffix. This original
    // create fills the window and triggers the failing body publication. Its
    // HTTP outcome can be unknown; the retained plan below proves the exact
    // original WAL identity that the successor must apply without resubmission.
    let body = fixture::encoded(fixture::create(&["U"], &["s"], &[2])?)?;
    let response = post(client, ingress, "update")
        .header(crate::http::PRODUCER, &producer.producer)
        .header(crate::http::SEQUENCE, 4)
        .header(reqwest::header::CONTENT_TYPE, crate::http::ARROW_FILE)
        .body(body)
        .send()
        .await;
    let completed = match response {
        Ok(response) if response.status().is_success() => {
            Some(response.json::<MutationReply>().await?.completion)
        }
        Ok(response) => {
            assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
            let value: serde_json::Value = response.json().await?;
            assert_eq!(value["outcome"], "unknown", "{value}");
            None
        }
        Err(error) => {
            ensure!(
                error.is_timeout() || error.is_request() || error.is_body(),
                "{error}"
            );
            None
        }
    };
    failed(owner).await?;
    assert_eq!(owner.0.try_wait()?.unwrap().code(), Some(1));
    let log = std::fs::read_to_string(root.join("owner.log"))?;
    ensure!(
        log.contains("Permission denied"),
        "owner failed for another reason: {log}"
    );
    let failed = annotation_publication::state(c).await?;
    assert_eq!(failed.authority, before.authority);
    assert_eq!(failed.manifests, before.manifests);
    assert!(failed.progress.prepared.is_none());
    let pending = failed
        .progress
        .pending
        .as_ref()
        .context("failed publication plan missing")?;
    let records = &pending.plan.sources[&10].records;
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].0, acknowledged);
    assert_eq!(records[0].1.sequence, 3);
    assert_eq!(records[1].1.sequence, 4);
    assert_eq!(records[0].1.producer, records[1].1.producer);
    if let Some(completed) = completed {
        assert_eq!(completed.index, records[1].0);
        assert_eq!(
            completed.result,
            crate::annotations::MutationResult::Applied
        );
    }
    for (index, _) in records {
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: *index
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    blocked.restore()?;
    Ok(records[1].0)
}
