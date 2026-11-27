use super::*;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::{dataset::WriteParams, Dataset};
use lance_context_ingest::{
    annotations::{single_body, Shard},
    application::config::{Rpc, TlsFiles, WalConnection},
    turn_ingestion::TurnIngestionSpec,
    TableBinding,
};
use lance_context_wal::{
    placement::Layout,
    replication::{
        publication::{Authority, Destination, ManifestObject, ManifestRecord},
        query::Cut,
    },
};
use lance_core::memory::{NativeMemoryKind, NativeMemoryPool};
use std::collections::BTreeMap;
use uuid::Uuid;

fn fixture(uri: String) -> Result<(Config, Capture)> {
    let layout = Layout {
        cluster: Uuid::new_v4(),
        nodes: (1..=3).map(|id| (id, format!("wal-{id}:7800"))).collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3])]),
        buckets: vec![10],
    };
    let binding = TableBinding {
        table: Uuid::new_v4(),
        incarnation: Uuid::new_v4(),
        destination: Destination {
            namespace: Uuid::new_v4(),
            dataset: "test".into(),
        },
        tenant: "test".into(),
        qualified_table: "turns".into(),
        max_record_bytes: 1 << 20,
        max_rows: 1024,
    };
    let spec = TurnIngestionSpec::new(
        Arc::new(Schema::new(vec![
            Field::new("session", DataType::Utf8, false),
            Field::new("id", DataType::Utf8, false),
        ])),
        vec!["session".into(), "id".into()],
        "session",
    )?;
    let staging = Default::default();
    let shard = Shard { bucket: 0 };
    let mut keys = vec![
        spec.business_key(&["session-0", "id-0"], 1024)?,
        spec.business_key(&["session-0", "id-1"], 1024)?,
        spec.business_key(&["session-0", "missing"], 1024)?,
    ];
    keys.sort();
    let capture = Capture {
        capture_id: Uuid::new_v4(),
        owner: Authority {
            cluster: layout.cluster,
            layout: layout.fingerprint()?,
            group: 10,
            table: binding.table,
            owner: Uuid::new_v4(),
            epoch: 1,
            grant_index: 1,
        },
        incarnation: binding.incarnation,
        schema: single_body::contract(&binding, &spec, &staging, shard)?.schema,
        source_group: 10,
        bucket: 0,
        base_version: 1,
        base_fragment_ids: vec![0, 1, 2, 3],
        window: input::Window {
            first_record: 10,
            last_record: 12,
            requests: 3,
            payload_bytes: 1024,
            speculative: false,
        },
        requested_keys: 3,
        staged_keys: 0,
        primary_key_columns: spec.primary_key().to_vec(),
        keys,
    };
    let config = Config {
        wal: WalConnection {
            layout,
            limits: Default::default(),
            endpoints: BTreeMap::new(),
            tls: TlsFiles {
                certificate: "unused".into(),
                private_key: "unused".into(),
                roots: "unused".into(),
            },
            rpc: Rpc::default(),
        },
        binding,
        shard,
        staging,
        spec,
        uri,
        storage_options: Default::default(),
    };
    Ok((config, capture))
}

// Test-only catalog identity fixture. Real records MUST come from record(),
// which performs authenticated selection; reconstructing identities is not auth.
fn selected(capture: &Capture, bytes: u64, transaction: Uuid) -> Record {
    let index = 17u64;
    let mut hash = ring::digest::Context::new(&ring::digest::SHA256);
    for bytes in [
        b"lance-context/manifest-identity/v1".as_slice(),
        capture.owner.cluster.as_bytes(),
        &capture.owner.layout,
        capture.owner.table.as_bytes(),
        capture.incarnation.as_bytes(),
        &capture.base_version.to_le_bytes(),
        &index.to_le_bytes(),
    ] {
        hash.update(bytes);
    }
    Record {
        capture_id: capture.capture_id,
        cluster: capture.owner.cluster,
        layout: capture.owner.layout,
        table: capture.owner.table,
        incarnation: capture.incarnation,
        schema: capture.schema,
        source_group: capture.source_group,
        bucket: capture.bucket,
        observed: Cut {
            cluster: capture.owner.cluster,
            layout: capture.owner.layout,
            group: 10,
            term: 1,
            quorum_index: index,
            applied: index,
        },
        manifest: ManifestRecord {
            version: capture.base_version,
            identity: Uuid::from_bytes(hash.finish().as_ref()[..16].try_into().unwrap()),
            object: ManifestObject {
                path: "_versions/1.manifest".into(),
                bytes,
                v2_names: false,
            },
            transaction,
            index,
            plan_index: 0,
            prepared_index: 0,
            descriptor_digest: [0; 32],
        },
    }
}

#[test]
fn capture_and_record_reject_foreign_scope_and_incomplete_keys() -> Result<()> {
    let (config, mut capture) = fixture("unused".into())?;
    config.validate_capture(&capture)?;
    let mut record = selected(&capture, 1024, Uuid::new_v4());
    record.validate(&config, &capture)?;
    capture.schema = config
        .shard
        .turn_contract(&config.binding, &config.spec, &config.staging)?
        .schema;
    assert!(config.validate_capture(&capture).is_err()); // Mixed wire is NOT full-body wire.
    capture.schema = record.schema;
    capture.requested_keys += 1;
    assert!(config.validate_capture(&capture).is_err());
    capture.requested_keys -= 1;
    capture.keys.swap(0, 1);
    assert!(config.validate_capture(&capture).is_err());
    capture.keys.swap(0, 1);
    record.incarnation = Uuid::new_v4();
    assert!(record.validate(&config, &capture).is_err());
    record.incarnation = capture.incarnation;
    record.manifest.version += 1;
    assert!(record.validate(&config, &capture).is_err());
    record.manifest.version -= 1;
    record.observed.applied = 16;
    assert!(record.validate(&config, &capture).is_err());
    Ok(())
}

#[test]
fn experimental_pool_releases_backing_and_refuses_without_waiting() -> Result<()> {
    let pool = pool::Pool::new(10);
    let mut charge = pool.try_reserve(NativeMemoryKind::Read, 8)?;
    charge.set_backed_bytes(7);
    assert_eq!(pool.snapshot().backed, 7);
    assert!(pool.try_reserve(NativeMemoryKind::Decode, 3).is_err());
    assert_eq!(pool.snapshot().refused, 1);
    drop(charge);
    assert_eq!(pool.snapshot().reserved, 0);
    assert_eq!(pool.snapshot().backed, 0);
    assert_eq!(pool.snapshot().by_kind, [0; 3]);
    Ok(())
}

#[tokio::test]
async fn pinned_fixture_compares_all_settings_and_reports_refusal_without_keys() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let (config, mut capture) = fixture(directory.path().join("table").to_str().unwrap().into())?;
    let schema = single_body::storage_schema(&config.spec)?;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["session-0"; 32])),
            Arc::new(StringArray::from_iter_values(
                (0..32).map(|n| format!("id-{n}")),
            )),
        ],
    )?;
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &config.uri,
        Some(WriteParams {
            max_rows_per_file: 8,
            enable_v2_manifest_paths: false,
            ..Default::default()
        }),
    )
    .await?;
    capture.base_fragment_ids = dataset.fragments().iter().map(|f| f.id).collect();
    assert_eq!(capture.base_fragment_ids.len(), 4);
    let record = selected(
        &capture,
        std::fs::metadata(directory.path().join("table/_versions/1.manifest"))?.len(),
        Uuid::parse_str(&dataset.read_transaction().await?.unwrap().uuid)?,
    );
    let pool = Arc::new(pool::Pool::new(256 << 20));
    let storage = lance_context_ingest::materializer::Storage::resolve(
        config.uri.clone(),
        Default::default(),
        Arc::new(lance::session::Session::default().with_native_memory(pool.clone())),
    )
    .await?;
    let base = probe::open(&config, &capture, &record, &storage).await?;
    capture.base_fragment_ids.reverse();
    assert!(probe::open(&config, &capture, &record, &storage)
        .await
        .is_err());
    capture.base_fragment_ids.reverse();
    let mut out = Vec::new();
    probe::compare(
        &config,
        &capture,
        base,
        probe::Options {
            key_workspace: lance_context_core::MergeMemoryBudget::new(1 << 20),
            rounds: 3,
            timeout: Duration::from_secs(30),
        },
        &mut out,
        || serde_json::to_value(pool.snapshot()).unwrap(),
    )
    .await?;
    let text = std::str::from_utf8(&out)?;
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    assert_eq!(lines.len(), 17);
    for row in &lines[..16] {
        assert_eq!(row["status"], "equal");
        assert_eq!(row["observation"]["history_hits"], 2);
        assert_eq!(row["observation"]["history_misses"], 1);
    }
    assert!(!text.contains("session-0") && !text.contains("id-0") && !text.contains("missing"));
    let refused = Arc::new(pool::Pool::new(0));
    let storage = lance_context_ingest::materializer::Storage::resolve(
        config.uri.clone(),
        Default::default(),
        Arc::new(lance::session::Session::default().with_native_memory(refused.clone())),
    )
    .await?;
    let base = probe::open(&config, &capture, &record, &storage).await?;
    let mut out = Vec::new();
    assert!(probe::compare(
        &config,
        &capture,
        base,
        probe::Options {
            key_workspace: lance_context_core::MergeMemoryBudget::new(1 << 20),
            rounds: 1,
            timeout: Duration::from_secs(30),
        },
        &mut out,
        || serde_json::to_value(refused.snapshot()).unwrap()
    )
    .await
    .is_err());
    let row: serde_json::Value = serde_json::from_slice(&out)?;
    assert_eq!(row["status"], "native_backpressure");
    assert!(row.get("observation").is_none());
    Ok(())
}

#[test]
fn private_inputs_and_atomic_record_outputs() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("record.json");
    let (_, capture) = fixture("unused".into())?;
    let record = selected(&capture, 1024, Uuid::new_v4());
    assert!(!path.exists());
    record_output(&path, &record)?;
    assert!(record_output(&path, &record).is_err());
    assert_eq!(input::read::<Record>(&path)?.manifest, record.manifest);
    assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path)?.permissions().mode() & 0o777,
            0o600
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))?;
        assert!(input::read::<Record>(&path).is_err());
    }
    Ok(())
}
