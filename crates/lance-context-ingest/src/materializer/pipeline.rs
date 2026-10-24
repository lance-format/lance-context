//! Bounded local streaming preparation within one immutable publication plan.
use super::*;
use lance::dataset::{transaction::Operation, InsertBuilder, WriteMode, WriteParams};
use lance::deps::datafusion::{
    error::DataFusionError,
    execution::SendableRecordBatchStream,
    physical_plan::stream::{RecordBatchReceiverStreamBuilder, RecordBatchStreamAdapter},
};
use tokio::sync::Semaphore;

pub(super) async fn materialize(
    ingress: Ingress,
    options: Options,
    inputs: Vec<(u64, u64, BatchMeta)>,
    dataset: Dataset,
    reservation: Arc<MergeReservation>,
    phases: Arc<PhaseCounters>,
) -> Result<p::Descriptors> {
    let writers = options.write_parallelism.min(inputs.len());
    ensure!(writers > 0, "empty pinned source selection");
    let mut partitions = (0..writers).map(|_| Vec::new()).collect::<Vec<_>>();
    // Identical record assignment and output order to the non-streaming path.
    for (i, record) in inputs.into_iter().enumerate() {
        partitions[i % writers].push(record);
    }
    let permits = Arc::new(Semaphore::new(options.fetch_parallelism));
    let policy = options.source_read_policy;
    let streams = partitions
        .into_iter()
        .map(|partition| {
            let ingress = ingress.clone();
            let schema = ingress.schema.clone();
            let permits = permits.clone();
            let reservation = reservation.clone();
            let phases = phases.clone();
            let records = stream::iter(partition.into_iter().map(move |(group, index, meta)| {
                let ingress = ingress.clone();
                let permits = permits.clone();
                let reservation = reservation.clone();
                let phases = phases.clone();
                async move {
                    let permit = permits.acquire_owned().await?;
                    super::stage::fetch(
                        ingress,
                        (group, index, meta),
                        policy,
                        reservation,
                        phases,
                        Some(permit),
                    )
                    .await
                }
            }))
            .buffered(options.fetch_parallelism)
            .map_err(|error| DataFusionError::External(error.into()))
            .map_ok(|batches| stream::iter(batches.into_iter().map(Ok)))
            .try_flatten();
            prefetch(
                Box::pin(RecordBatchStreamAdapter::new(schema, records)),
                options.fetch_parallelism,
            )
        })
        .collect();
    write_streams(dataset, streams, reservation, phases).await
}

fn prefetch(mut source: SendableRecordBatchStream, capacity: usize) -> SendableRecordBatchStream {
    // Poll WAL fetches independently while the file writer encodes or uploads.
    // The existing full-source reservation covers these queued Arrow batches;
    // sending them transfers shared buffers without another encode or decode.
    let mut output = RecordBatchReceiverStreamBuilder::new(source.schema(), capacity);
    let send = output.tx();
    output.spawn(async move {
        while let Some(batch) = source.try_next().await? {
            if send.send(Ok(batch)).await.is_err() {
                break;
            }
        }
        Ok(())
    });
    // DataFusion propagates task errors/panics and cancels the producer when
    // its receiver is dropped. A previously started blocking decode retains
    // its own reservation and fetch permit until it actually finishes.
    output.build()
}

async fn write_streams(
    dataset: Dataset,
    streams: Vec<SendableRecordBatchStream>,
    reservation: Arc<MergeReservation>,
    phases: Arc<PhaseCounters>,
) -> Result<p::Descriptors> {
    let dataset = Arc::new(dataset);
    let tasks = streams.into_iter().map(|source| {
        let dataset = dataset.clone();
        let phases = phases.clone();
        let reservation = reservation.clone();
        async move {
            let _reservation = reservation;
            // This enclosing phase includes time waiting for source records;
            // fetch/decode counters overlap it and must not be added to it.
            let staged = phases
                .measure(
                    Phase::BlobWrite,
                    InsertBuilder::new(dataset)
                        .with_params(&WriteParams {
                            mode: WriteMode::Append,
                            ..Default::default()
                        })
                        .execute_uncommitted_stream(source),
                )
                .await?;
            let Operation::Append { fragments } = staged.operation else {
                bail!("materialization did not produce append fragments");
            };
            Ok::<_, anyhow::Error>(fragments)
        }
    });
    // A failed stream never publishes a partial descriptor set. Keep the
    // source reservation until every sibling writer has exited.
    let results = futures::future::join_all(tasks).await;
    let fragments = results
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    p::Descriptors::lance_fragments(Bytes::from(serde_json::to_vec(&fragments)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatchIterator};
    use tokio::sync::{oneshot, Notify};

    #[tokio::test]
    async fn prefetch_advances_while_writer_is_idle_and_preserves_order() -> Result<()> {
        let batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int32Array::from(vec![0])) as arrow_array::ArrayRef,
        )])?;
        let schema = batch.schema();
        let (progress, mut observed) = tokio::sync::watch::channel(0);
        let source = async_stream::try_stream! {
            for value in 0..10 {
                progress.send_replace(value + 1);
                yield RecordBatch::try_from_iter(vec![(
                    "value",
                    Arc::new(Int32Array::from(vec![value])) as arrow_array::ArrayRef,
                )]).map_err(DataFusionError::from)?;
            }
        };
        let output = prefetch(Box::pin(RecordBatchStreamAdapter::new(schema, source)), 2);
        // The writer has not polled its input. Two batches may be queued and
        // one held by the sender; no further input should be consumed yet.
        tokio::time::timeout(Duration::from_secs(3), observed.wait_for(|n| *n >= 3))
            .await
            .context("source did not advance independently of the writer")??;
        tokio::task::yield_now().await;
        assert_eq!(*observed.borrow(), 3, "prefetch exceeded its queue bound");
        let batches = output.try_collect::<Vec<_>>().await?;
        let values = batches
            .iter()
            .map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .value(0)
            })
            .collect::<Vec<_>>();
        assert_eq!(values, (0..10).collect::<Vec<_>>());
        Ok(())
    }

    #[tokio::test]
    async fn dropping_prefetch_cancels_pending_source_and_releases_reservation() -> Result<()> {
        let batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int32Array::from(vec![1])) as arrow_array::ArrayRef,
        )])?;
        let schema = batch.schema();
        let budget = MergeMemoryBudget::new(1 << 20);
        let owned = budget.clone();
        let (entered, ready) = oneshot::channel();
        let source = async_stream::try_stream! {
            let _reservation = owned.reserve(1 << 20).await;
            let _ = entered.send(());
            std::future::pending::<()>().await;
            yield batch;
        };
        let output = prefetch(Box::pin(RecordBatchStreamAdapter::new(schema, source)), 1);
        tokio::time::timeout(Duration::from_secs(3), ready).await??;
        assert_eq!(budget.reserved(), 1 << 20);
        drop(output);
        let reservation =
            tokio::time::timeout(Duration::from_secs(3), budget.reserve(1 << 20)).await?;
        drop(reservation);
        assert_eq!(budget.reserved(), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn failed_stream_and_cancelled_observer_keep_sibling_and_budget_alive() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("streamed.lance");
        let batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int32Array::from(vec![1, 2, 3])) as arrow_array::ArrayRef,
        )])?;
        let schema = batch.schema();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
            path.to_str().unwrap(),
            None,
        )
        .await?;
        let initial_version = dataset.version().version;
        let failed = Arc::new(Notify::new());
        let blocked = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let first = {
            let failed = failed.clone();
            stream::iter(vec![Ok(batch.clone())]).chain(stream::once(async move {
                failed.notify_one();
                Err(DataFusionError::Execution("injected source failure".into()))
            }))
        };
        let second = {
            let blocked = blocked.clone();
            let release = release.clone();
            stream::iter(vec![Ok(batch.clone())]).chain(stream::once(async move {
                blocked.notify_one();
                release.notified().await;
                Ok(batch)
            }))
        };
        let streams: Vec<SendableRecordBatchStream> = vec![
            Box::pin(RecordBatchStreamAdapter::new(schema.clone(), first)),
            Box::pin(RecordBatchStreamAdapter::new(schema, second)),
        ];
        let streams = streams
            .into_iter()
            .map(|stream| prefetch(stream, 1))
            .collect();
        let budget = MergeMemoryBudget::new(1 << 20);
        let reservation = Arc::new(budget.reserve(1 << 20).await);
        let (done, completion) = oneshot::channel();
        let job = tokio::spawn(async move {
            let result = write_streams(dataset, streams, reservation, Arc::default()).await;
            let _ = done.send(result.is_err());
            result
        });
        // This is the same ownership boundary as materialize: cancellation of
        // the observer drops a JoinHandle, while the admitted job keeps running.
        let observer = tokio::spawn(job);
        tokio::time::timeout(Duration::from_secs(10), async {
            failed.notified().await;
            blocked.notified().await;
        })
        .await?;
        assert!(
            !observer.is_finished(),
            "failed writer cancelled its sibling"
        );
        assert_eq!(budget.reserved(), 1 << 20);
        observer.abort();
        assert!(observer.await.unwrap_err().is_cancelled());
        assert_eq!(budget.reserved(), 1 << 20);
        release.notify_one();
        assert!(tokio::time::timeout(Duration::from_secs(10), completion).await??);
        assert_eq!(budget.reserved(), 0);
        let reopened = Dataset::open(path.to_str().unwrap()).await?;
        assert_eq!(reopened.version().version, initial_version);
        assert_eq!(reopened.count_rows(None).await?, 3);
        Ok(())
    }
}
