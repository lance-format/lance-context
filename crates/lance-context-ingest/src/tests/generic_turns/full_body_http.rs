//! Single full-body Turns through the real HTTP stack: owner server, two
//! forwarding ingresses, the durable producer journal, producer replacement
//! and selection. Payloads are the original application rows.
use super::http::{keys, open_client, transport, Server};
use super::*;
use crate::annotations::{
    runtime::{Configuration, Runtime},
    single_body,
};
use crate::http::{
    annotations::{Forwarder, Owner},
    journal::{ProducerControlClient, ReplacementIntent},
};
use crate::materializer::{Acquisition, Options};
use crate::tests::annotation_runtime::{drained, run_owner};

fn spec(input: &[RecordBatch]) -> Result<TurnIngestionSpec> {
    TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )
}

fn open(slot: u32, generation: u64, operation: u128) -> ProducerOpen {
    ProducerOpen {
        bucket: 0,
        slot,
        expected_generation: generation,
        operation: Uuid::from_u128(operation),
    }
}

async fn owner(
    c: &network::Cluster,
    storage: &Storage,
    spec: &TurnIngestionSpec,
) -> Result<(Service, Runtime)> {
    let view = annotation_publication::state(c).await?;
    Runtime::connect_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: fixture::limits(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 2,
                max_batch_wait: Duration::from_millis(20),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: view.authority.epoch,
        },
        Shard { bucket: 0 },
        spec,
    )
    .await
}

/// A separate journal parent per producer executor; it must already exist.
fn executor(root: &std::path::Path, name: &str) -> Result<std::path::PathBuf> {
    let path = root.join(name);
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

fn wire(spec: &TurnIngestionSpec, body: &RecordBatch) -> Result<Bytes> {
    fixture::encoded(single_body::batch(
        spec,
        &fixture::limits(),
        Shard { bucket: 0 },
        body,
    )?)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_http_client_dedups_across_producers_reopens_and_selects() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(120), exercise()).await?
}

async fn exercise() -> Result<()> {
    let input = bodies()?;
    let spec = spec(&input)?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let body = fixture::storage(root.path(), "full-body.lance").await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        body.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let (service, runtime) = owner(&c, &body, &spec).await?;
    assert!(service.ingress().turn_spec.is_some());
    let server = Server::start(Owner::new(&runtime, keys()?, Default::default())?.router()).await?;
    let (stop, task) = run_owner(runtime);
    let forward = || {
        Forwarder::new(
            vec![(service.ingress().clone(), server.base.to_string())],
            keys()?,
            Default::default(),
            "test-only-token-012345678901234567890".into(),
        )
    };
    let a = Server::start(forward()?.router()).await?;
    let b = Server::start(forward()?.router()).await?;

    // Producer one writes the first two original bodies through ingress A.
    let one = executor(root.path(), "one")?;
    let writer = open_client(&one, a.base.clone(), &spec, open(1, 0, 101), true).await?;
    let first = writer.append(0, wire(&spec, &input[0])?).await?;
    assert_eq!(first.completion.result, MutationResult::Applied);
    assert_eq!(first.completion.meta.sequence, 1);
    let second = writer.append(0, wire(&spec, &input[1])?).await?;
    assert_eq!(second.completion.meta.sequence, 2);
    writer.shutdown().await?;

    // Reopen the same journal through the other ingress: no pending intent,
    // and the last terminal result is retained exactly.
    let writer = open_client(&one, b.base.clone(), &spec, open(1, 0, 101), true).await?;
    assert!(writer.resume(0).await?.is_none());
    assert_eq!(writer.last_completed(0).await?, Some(second.clone()));

    // A second, independent producer repeats body 0 and adds body 3. The
    // repeated business keys are absorbed by first-create, not sequence.
    let two = executor(root.path(), "two")?;
    let other = open_client(&two, a.base.clone(), &spec, open(2, 0, 202), true).await?;
    let repeated = other.append(0, wire(&spec, &input[0])?).await?;
    assert_eq!(repeated.completion.result, MutationResult::Applied);
    assert_eq!(repeated.completion.meta.sequence, 1);
    let added = other.append(0, wire(&spec, &input[3])?).await?;
    drained(&c, &service, added.completion.index).await?;

    // Selection through HTTP equals the owner's own selection and opens the
    // exact manifest version, which holds each business key exactly once.
    let selected = writer.selected_full_body().await?;
    let direct = service.selected_full_body(DEADLINE).await?;
    assert_eq!(selected.uri, direct.uri);
    assert_eq!(selected.manifest.version, direct.manifest.version);
    assert_eq!(selected.incarnation, binding().incarnation);
    assert!(selected.through >= added.completion.index);
    let durable = annotation_publication::read(&c, &body, Some(selected.manifest.version)).await?;
    let keys = |i: usize| {
        spec.prepare(input[i].clone(), 4096)
            .map(|p| p.keys().to_vec())
    };
    // The fixture's bodies overlap in business keys; count distinct keys.
    let distinct = |bodies: &[usize]| -> Result<usize> {
        let mut all = std::collections::BTreeSet::new();
        for i in bodies {
            all.extend(keys(*i)?);
        }
        Ok(all.len())
    };
    let expected = distinct(&[0, 1, 3])?;
    assert_eq!(durable.count_rows(None).await?, expected);
    // Each key returns the body from the first body that created it.
    let mut first = std::collections::BTreeMap::new();
    for i in [0, 1, 3] {
        for (row, key) in keys(i)?.into_iter().enumerate() {
            first.entry(key).or_insert(input[i].slice(row, 1));
        }
    }
    for (key, original) in &first {
        let found = service
            .get_full_body(key, DEADLINE)
            .await?
            .context("missing body")?;
        assert_eq!(found.batch(), original, "first-created body changed");
    }
    // A mixed-mode client cannot be served by this single-body owner.
    let mixed = open_client(
        &executor(root.path(), "mixed")?,
        a.base.clone(),
        &spec,
        open(3, 0, 303),
        false,
    )
    .await?;
    assert!(mixed.selected_body().await.is_err());
    mixed.shutdown().await?;

    // Producer replacement uses the single-body scope and the existing state
    // machine: the successor continues the lane at the next generation.
    let control = ProducerControlClient::new_full_body(
        server.base.clone(),
        transport()?,
        DEADLINE,
        &annotation_publication::one_shard(),
        &binding(),
        &spec,
        &fixture::limits(),
        Shard { bucket: 0 },
    )?;
    let mixed_control = ProducerControlClient::new(
        server.base.clone(),
        transport()?,
        DEADLINE,
        &annotation_publication::one_shard(),
        &binding(),
        &spec,
        &fixture::limits(),
        Shard { bucket: 0 },
    )?;
    let predecessor = open(2, 0, 202);
    let intent: ReplacementIntent = control
        .prepare_replacement(&predecessor, Uuid::from_u128(909))
        .await?;
    assert!(
        mixed_control.execute(&intent).await.is_err(),
        "a mixed-scope control client must not replace a single-body producer"
    );
    // The predecessor's tail is already applied and published, so the
    // replacement opens once retirement catches up; waiting is explicit.
    let successor = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match control.execute(&intent).await? {
                crate::http::annotations::ReplaceProducerReply::Opened { open, .. } => {
                    return Ok::<_, anyhow::Error>(open)
                }
                crate::http::annotations::ReplaceProducerReply::WaitingForRetirement { .. } => {
                    tokio::time::sleep(Duration::from_millis(20)).await
                }
            }
        }
    })
    .await??;
    assert_eq!(
        successor.expected_generation,
        predecessor.expected_generation + 1
    );
    other.shutdown().await?;
    let next = open_client(
        &executor(root.path(), "three")?,
        b.base.clone(),
        &spec,
        successor,
        true,
    )
    .await?;
    let after = next.append(0, wire(&spec, &input[2])?).await?;
    assert_eq!(after.completion.result, MutationResult::Applied);
    drained(&c, &service, after.completion.index).await?;
    let final_selected = next.selected_full_body().await?;
    let durable =
        annotation_publication::read(&c, &body, Some(final_selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, distinct(&[0, 1, 2, 3])?);

    next.shutdown().await?;
    writer.shutdown().await?;
    for s in [a, b, server] {
        s.shutdown().await?;
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

/// Application config: single-body mode (turns set, no mixed body) validates
/// without a separate body destination and rejects extra-table options.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_body_application_config_requires_one_table_and_a_shard() -> Result<()> {
    use crate::application::config;
    let input = bodies()?;
    let spec = spec(&input)?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let mut base = c.application.clone();
    base.tables = vec![config::Table {
        binding: binding(),
        uri: root.path().join("turns.lance").to_str().unwrap().to_owned(),
        storage_options: Default::default(),
        bootstrap: Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
    }];
    let annotations = |turns: Option<TurnIngestionSpec>| config::Annotations {
        turns,
        owner_url: "http://127.0.0.1:1/".into(),
        shard: Some(Shard { bucket: 0 }),
        routes: vec![],
        wal_clusters: Default::default(),
        apply_memory_bytes: 8 << 20,
        index_cache_bytes: None,
        staging: fixture::limits(),
        source: None,
        projection: None,
        body: None,
        exclusive_reclamation: None,
    };
    let with = |a: config::Annotations| {
        let mut config = base.clone();
        config.annotations = Some(a);
        config
    };
    let single = annotations(Some(spec.clone()));
    with(single.clone()).validate()?;
    // No shard: rejected.
    let mut no_shard = single.clone();
    no_shard.shard = None;
    assert!(with(no_shard).validate().is_err());
    // A separate mixed body makes it the old mixed mode, not single-body.
    let mut mixed = single.clone();
    mixed.body = Some(config::Body {
        uri: root.path().join("body.lance").to_str().unwrap().to_owned(),
        storage_options: Default::default(),
    });
    with(mixed).validate()?;
    // Without turns, an explicit shard still needs mixed body storage.
    assert!(with(annotations(None)).validate().is_err());
    // Two local tables are rejected for any annotation application.
    let mut two = with(single);
    let mut extra = two.tables[0].clone();
    extra.binding.table = Uuid::new_v4();
    extra.binding.qualified_table = "other".into();
    extra.uri = root.path().join("other.lance").to_str().unwrap().to_owned();
    two.tables.push(extra);
    assert!(two.validate().is_err());
    c.shutdown().await
}
