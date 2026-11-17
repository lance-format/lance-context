//! Real RF3 preparation and detached binary-copy rewrite, without a mixed runtime.
use super::*;
use lance::dataset::optimize::{CompactionMode, CompactionOptions, CompactionTask, TaskData};
use lance_table::transaction::RewriteGroup;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paired_body_compaction_recovers_prepared_reference() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), super::recover(false, Some(0))).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paired_body_compaction_recovers_unknown_selection() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), super::recover(false, Some(1))).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paired_body_compaction_recovers_confirmed_selection() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), super::recover(false, Some(2))).await?
}

async fn bodies(dataset: &Dataset) -> Result<Vec<Vec<u8>>> {
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .project(&["binary_payload"])?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    Ok(batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap()
                .iter()
                .map(|value| value.unwrap().to_vec())
        })
        .collect())
}

async fn later_update(c: &network::Cluster) -> Result<(u64, BatchMeta)> {
    let schema = patch_schema();
    let rows = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["T", "U"])),
            Arc::new(StringArray::from(vec![Some("later"); 2])),
            Arc::new(Float32Array::from(vec![None; 2])),
        ],
    )?;
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &schema)?;
    writer.write(&rows)?;
    writer.finish()?;
    drop(writer);
    let batch = Batch::new(
        binding().table,
        0,
        producer::producer_id(0, 1),
        3,
        bytes.into(),
    );
    let meta = batch.meta.clone();
    let response = c
        .client
        .request_routed(Request::Append { group: 10, batch })
        .await?;
    let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = response.reply else {
        bail!("later update was not committed");
    };
    Ok((index, meta))
}

pub(super) async fn recover(
    root: &std::path::Path,
    c: network::Cluster,
    narrow: &Storage,
    body: &Storage,
    staged: AnnotationStaging,
    suffix: &(u64, BatchMeta),
    stage: u8,
) -> Result<()> {
    let base = read(&c, narrow, None).await?;
    let covered = staged.observation().await.materialized;
    let old_reference = BodySnapshot::read(&base).await?.unwrap();
    let original = DatasetBuilder::from_uri(&body.uri)
        .with_version(old_reference.version)
        .load()
        .await?;
    assert_eq!(original.manifest().fragments.len(), 2);
    let expected_bodies = bodies(&original).await?;
    let held = staged.capture_turn("T").await?;
    let before = held.annotations().resolve().await?.unwrap();
    let view = state(&c).await?;
    let plan = p::Plan {
        transaction: Uuid::new_v4(),
        read_version: base.version_id(),
        sources: BTreeMap::new(),
        next_group: 10,
    };
    let Applied::Publisher(p::Reply::Planned { index }) = command(
        &c,
        Command::Plan(p::PlanRequest {
            authority: view.authority,
            plan: plan.clone(),
        }),
    )
    .await?
    else {
        bail!("maintenance plan rejected");
    };
    let selection = annotations::Selection {
        table: binding().table,
        incarnation: binding().incarnation,
        pending: p::PendingPlan { index, plan },
    };
    let task = CompactionTask {
        task: TaskData {
            fragments: original.manifest().fragments.as_ref().clone(),
        },
        read_version: original.version_id(),
        options: CompactionOptions {
            target_rows_per_fragment: 2,
            max_bytes_per_file: Some(1 << 20),
            compaction_mode: Some(CompactionMode::TryBinaryCopy),
            ..Default::default()
        },
    };
    let mut rewrite = task.execute(&original).await?;
    assert_eq!(rewrite.new_fragments.len(), 1);
    for (ordinal, fragment) in rewrite.new_fragments.iter_mut().enumerate() {
        fragment.id = u64::from(original.manifest().max_fragment_id.unwrap()) + 1 + ordinal as u64;
    }
    let mut transaction = Transaction::new_from_version(
        original.version_id(),
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: rewrite.original_fragments,
                new_fragments: rewrite.new_fragments,
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    );
    transaction.uuid = selection.pending.plan.transaction.to_string();
    transaction.transaction_properties = Some(Arc::new(HashMap::from([(
        "lance-context.annotation-cut".into(),
        serde_json::to_string(&(scope(), covered))?,
    )])));
    let rewritten = CommitBuilder::new(Arc::new(original))
        .with_detached(true)
        .with_max_retries(1)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    let reference = BodySnapshot::capture(binding().incarnation.to_string(), &rewritten).await?;
    let descriptors = annotations::mixed::maintenance::descriptors(
        &base,
        reference.clone(),
        &selection,
        narrow,
        body,
        512,
    )
    .await?;
    assert_eq!(descriptors.format, 5);
    let commit = save(&c, &selection, descriptors.clone()).await?;
    let later = later_update(&c).await?;
    apply(&c, &staged, &later).await?;
    let latest = staged.capture("T").await?.resolve().await?.unwrap();
    assert_eq!(latest.status.as_deref(), Some("later"));
    assert_eq!(latest.score, Some(0.9));
    if stage > 0 {
        assert!(annotations::mixed::publish(
            Arc::new(LostReply::new(Arc::new(c.client.manifest_catalog()))),
            &one_shard(),
            narrow,
            body,
            commit.clone(),
            &selection,
            &descriptors,
            base.clone(),
        )
        .await
        .is_err());
        let selected = read(&c, narrow, None).await?;
        let transaction = selected.read_transaction().await?.unwrap();
        if stage == 2 {
            confirm(&c).await?;
        }
        // Install only after a confirmed publication, preserving the active suffix.
        if stage == 2 {
            staged
                .install_body_compacted(&transaction, selected)
                .await?;
            assert_eq!(staged.capture("T").await?.resolve().await?.unwrap(), latest);
            let lance_context_core::annotation_staging::TurnBodySource::Materialized(old) =
                held.body_source()
            else {
                bail!("held reader lost its original base");
            };
            assert_eq!(BodySnapshot::read(old).await?, Some(old_reference));
            assert_eq!(held.annotations().resolve().await?, Some(before));
        }
    }
    let saved = state(&c).await?.progress.prepared.as_ref().unwrap().clone();
    drop((staged, held, rewritten, base));
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root, one_shard()).await?;
    // Reopen the original version with the restarted cluster's catalog client.
    // The pre-restart Dataset retains the old managed handler and endpoints.
    let base = read(&c, narrow, Some(selection.pending.plan.read_version)).await?;
    let recovered = state(&c).await?;
    assert_eq!(recovered.progress.prepared.as_ref().unwrap(), &saved);
    let selected = if stage == 0 {
        Arc::new(
            annotations::mixed::publish(
                Arc::new(c.client.manifest_catalog()),
                &one_shard(),
                narrow,
                body,
                commit,
                &selection,
                &descriptors,
                base.clone(),
            )
            .await?,
        )
    } else {
        read(&c, narrow, None).await?
    };
    let view = state(&c).await?;
    assert_eq!(
        annotations::mixed::selected_cut(
            &selected,
            &base,
            &descriptors,
            &selection,
            &view.manifests.as_ref().unwrap().versions[&selected.version_id()],
            narrow,
            body,
        )
        .await?,
        covered
    );
    let prior_tx = base.read_transaction().await?.unwrap();
    let selected_tx = selected.read_transaction().await?.unwrap();
    assert_eq!(
        prior_tx
            .transaction_properties
            .as_ref()
            .unwrap()
            .get(lance_context_core::annotation_staging::RESULTS_PROPERTY),
        selected_tx
            .transaction_properties
            .as_ref()
            .unwrap()
            .get(lance_context_core::annotation_staging::RESULTS_PROPERTY),
    );
    assert_eq!(
        BodySnapshot::read(&selected).await?,
        Some(reference.clone())
    );
    let recovered_staging = staging(selected, covered)?;
    apply(&c, &recovered_staging, suffix).await?;
    apply(&c, &recovered_staging, &later).await?;
    assert_eq!(
        recovered_staging
            .capture("T")
            .await?
            .resolve()
            .await?
            .unwrap(),
        latest
    );
    let body_after = DatasetBuilder::from_uri(&body.uri)
        .with_version(reference.version)
        .load()
        .await?;
    assert_eq!(body_after.manifest().fragments.len(), 1);
    assert_eq!(bodies(&body_after).await?, expected_bodies);
    let published = confirm(&c).await?;
    let view = state(&c).await?;
    assert!(matches!(
        command(
            &c,
            Command::Advance(p::AdvancePublication {
                authority: view.authority,
                published_index: published.index,
                proofs: BTreeMap::new(),
            })
        )
        .await?,
        Applied::Publisher(p::Reply::Advanced(_))
    ));
    assert_eq!(
        state(&c).await?.progress.checkpoint.get(&10),
        Some(&covered)
    );
    for record in [suffix, &later] {
        let response = c
            .client
            .request_routed(Request::Fetch {
                group: 10,
                index: record.0,
            })
            .await?;
        assert!(matches!(response.reply, Reply::Record(Some(meta)) if meta == record.1));
    }
    c.shutdown().await
}
