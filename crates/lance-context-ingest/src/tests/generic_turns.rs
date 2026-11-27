//! Exercise the existing RF3/apply/Lance path with an arbitrary body schema.
use super::*;
use super::{
    annotations::{producer, request, run},
    mixed_annotations as fixture,
};
use crate::{
    annotations::{mixed, MutationResult, Service, Shard},
    bootstrap::{annotations::mixed::provision_turn_shard, Identity},
    materializer::Storage,
    turn_ingestion::TurnIngestionSpec,
};
use arrow_array::Float64Array;
use arrow_schema::{DataType, Field, Schema};
use std::collections::HashMap;

const DEADLINE: Duration = Duration::from_secs(30);

#[cfg(all(feature = "http-client", feature = "service"))]
mod http;
mod single_body;

fn bodies() -> Result<Vec<RecordBatch>> {
    // An operator may repeat this exact product-path test with externally
    // generated Arrow files, without putting private fixtures in the repository.
    if let Some(directory) = std::env::var_os("LANCE_CONTEXT_TURN_FIXTURE_DIR") {
        return (0..4)
            .map(|i| {
                let file = std::fs::File::open(
                    std::path::Path::new(&directory).join(format!("request-{i}.arrow")),
                )?;
                let mut reader = arrow_ipc::reader::FileReader::try_new(file, None)?;
                let batch = reader.next().context("empty input fixture")??;
                ensure!(
                    reader.next().is_none(),
                    "fixture expects one batch per request"
                );
                Ok(batch)
            })
            .collect();
    }
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("session_id", DataType::Utf8, false),
            Field::new("id", DataType::Utf8, false),
            Field::new("parent_id", DataType::Utf8, true),
            Field::new("content", DataType::LargeBinary, false).with_metadata(HashMap::from([(
                "lance-encoding:compression".into(),
                "zstd".into(),
            )])),
            Field::new("score", DataType::Float64, true),
        ],
        HashMap::from([("projector".into(), "custom-v1".into())]),
    ));
    [
        ("s1", vec!["a", "b"]),
        ("s1", vec!["a", "b"]),
        ("s1", vec!["a", "b", "c", "d"]),
        ("s2", vec!["a", "b"]),
    ]
    .into_iter()
    .enumerate()
    .map(|(request, (session, ids))| {
        let rows = ids.len();
        Ok(RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![session; rows])),
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(vec![None::<&str>; rows])),
                Arc::new(LargeBinaryArray::from_iter_values(
                    (0..rows).map(|_| b"original body".as_slice()),
                )),
                Arc::new(Float64Array::from(vec![0.125 + request as f64; rows])),
            ],
        )?)
    })
    .collect()
}

async fn connect(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    spec: &TurnIngestionSpec,
) -> Result<(Service, crate::annotations::Driver)> {
    Service::connect_turn_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        narrow,
        body.clone(),
        fixture::limits(),
        MergeMemoryBudget::new(8 << 20),
        MergeMemoryBudget::new(8 << 20),
        Shard { bucket: 0 },
        spec,
    )
    .await
}

fn encode(batch: RecordBatch) -> Result<Bytes> {
    fixture::encoded(Shard { bucket: 0 }.bind_batch(&batch)?)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn generic_schema_deduplicates_preserves_annotations_and_recovers() -> Result<()> {
    let bodies = bodies()?;
    let spec = TurnIngestionSpec::new(
        bodies[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let keys = bodies
        .iter()
        .map(|b| spec.prepare(b.clone(), 1024).map(|p| p.keys().to_vec()))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        bodies.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
        [2, 2, 4, 2]
    );
    assert_eq!(keys.iter().flatten().collect::<HashSet<_>>().len(), 6);
    let output = std::env::var_os("LANCE_CONTEXT_TURN_OUTPUT_DIR");
    let root = if let Some(directory) = &output {
        std::fs::create_dir_all(directory)?;
        tempfile::tempdir_in(directory)?
    } else {
        tempfile::tempdir()?
    };
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = fixture::storage(root.path(), "generic-annotations.lance").await?;
    let body = fixture::storage(root.path(), "generic-body.lance").await?;
    provision_turn_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        narrow.clone(),
        body.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        fixture::limits(),
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let (a, driver) = connect(&c, &narrow, &body, &spec).await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 1).await?;
    let q = producer(&a, 2).await?;
    let creates = |i: usize| -> Result<Bytes> {
        let ids = keys[i].iter().map(String::as_str).collect::<Vec<_>>();
        let initial = fixture::patch(&ids, Some("pending"), None)?;
        encode(mixed::wire::creates_for(
            &spec,
            &bodies[i],
            &initial,
            &fixture::limits(),
        )?)
    };
    let first = request(&a, &p, 1, creates(0)?).await?;
    assert_eq!(
        a.submit(&first, DEADLINE).await?.result,
        MutationResult::Applied
    );
    let before = a.get_turn(&keys[0][0], DEADLINE).await?.unwrap();
    assert_eq!(before.body(), &bodies[0].slice(0, 1));
    drop(before);
    // Freeze first bodies, then update the same turn before publication.
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    let first_cut = frozen.through();
    let update = request(
        &a,
        &p,
        2,
        encode(mixed::wire::patches_for(
            &spec,
            &fixture::patch(&[&keys[0][0]], Some("reviewed"), Some(0.9))?,
            &fixture::limits(),
        )?)?,
    )
    .await?;
    assert_eq!(
        a.submit(&update, DEADLINE).await?.result,
        MutationResult::Applied
    );
    // A different producer repeats the body while both generations exist.
    let repeat = request(&a, &q, 1, creates(1)?).await?;
    assert_eq!(
        a.submit(&repeat, DEADLINE).await?.result,
        MutationResult::Applied
    );
    fixture::pressure::publish_frozen(&c, &a, &narrow, &body, frozen).await?;
    let selected = a.selected_body(DEADLINE).await?;
    assert_eq!(selected.through, first_cut);
    for (i, sequence) in [(2, 2), (3, 3)] {
        let create = request(&a, &q, sequence, creates(i)?).await?;
        assert_eq!(
            a.submit(&create, DEADLINE).await?.result,
            MutationResult::Applied
        );
    }
    fixture::pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    let selected = a.selected_body(DEADLINE).await?;
    let durable = lance::dataset::builder::DatasetBuilder::from_uri(&selected.body.uri)
        .with_version(selected.body.version)
        .load()
        .await?;
    assert_eq!(durable.count_rows(None).await?, 6);
    assert_eq!(Schema::from(durable.schema()), *spec.body_schema().as_ref());
    let current = a.get_turn(&keys[0][0], DEADLINE).await?.unwrap();
    assert_eq!(current.body(), &bodies[0].slice(0, 1));
    assert_eq!(current.annotation.status.as_deref(), Some("reviewed"));
    assert_eq!(current.annotation.score, Some(0.9));
    drop(current);
    let other = a.get_turn(&keys[3][0], DEADLINE).await?.unwrap();
    assert_eq!(other.body(), &bodies[3].slice(0, 1));
    assert_eq!(other.annotation.status.as_deref(), Some("pending"));
    drop(other);
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    // Rebuild solely from the selected Lance snapshot and durable WAL suffix.
    let (reopened, driver) = connect(&c, &narrow, &body, &spec).await?;
    let (stop, task) = run(driver);
    let current = reopened.get_turn(&keys[0][0], DEADLINE).await?.unwrap();
    assert_eq!(current.body(), &bodies[0].slice(0, 1));
    assert_eq!(current.annotation.score, Some(0.9));
    drop(current);
    let repeated = request(&reopened, &p, 3, creates(0)?).await?;
    assert_eq!(
        reopened.submit(&repeated, DEADLINE).await?.result,
        MutationResult::Applied
    );
    fixture::pressure::publish(&c, &reopened, &narrow, &body).await?;
    let after = reopened.selected_body(DEADLINE).await?;
    assert_eq!(
        selected.body, after.body,
        "zero-new-turn publication must not write new bodies"
    );
    assert!(
        after.through > selected.through,
        "duplicate-only source progress must still publish"
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    if output.is_some() {
        std::fs::write(
            root.path().join("turn-spec.json"),
            serde_json::to_vec_pretty(&spec)?,
        )?;
        std::fs::write(
            root.path().join("selected-body.json"),
            serde_json::to_vec_pretty(&after)?,
        )?;
        eprintln!("retained managed-turn artifacts: {}", root.keep().display());
    }
    Ok(())
}
