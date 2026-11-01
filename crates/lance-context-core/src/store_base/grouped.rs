//! Opt-in grouping of prepared shards into one atomic base update.
//!
//! Input order is commit order: a later shard wins for a repeated key, just as
//! it would when the existing ordered publisher commits the shards separately.
//! Generation numbers are local to a shard and must not order different shards.
use super::*;

// Only independent shard metadata operations overlap; base publication stays atomic.
const METADATA_CONCURRENCY: usize = 8;

/// An opaque, budgeted shard preparation bound to its source table.
pub struct PreparedShardMerge {
    uri: String,
    key_column: String,
    manifest_store: ShardManifestStore,
    manifest: ShardManifest,
    prepared: PreparedMerge,
    parallel_encoding: bool,
}

impl PreparedShardMerge {
    /// Opt in to two immutable encoders after group-wide key reduction.
    /// Publication remains atomic; insufficient workspace retains serial encoding.
    pub fn use_parallel_group_encoding(&mut self) {
        self.parallel_encoding = true;
    }

    /// Prepare immutable shard data under durable maintenance ownership, join
    /// its native readers, then retain only the resulting Arrow buffer charge.
    /// The callback must perform all reads inside this scope and must not
    /// publish. The returned context typically owns the publishing store.
    /// Cancellation retains the full reservation until the owned child joins;
    /// unused decode allowance is never released by an unjoined preparation.
    pub async fn prepare_owned<T, F, Fut>(
        reservation: MergeReservation,
        prepare: F,
    ) -> LanceResult<(T, Option<Self>)>
    where
        T: Send + 'static,
        F: FnOnce(MergeReservation) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = LanceResult<(T, Option<Self>)>> + Send + 'static,
    {
        let hold = reservation.retain();
        let (context, mut shard) =
            crate::merge_write_scope::run_owned_preparation(hold, async move {
                prepare(reservation).await
            })
            .await??;
        if let Some(shard) = &mut shard {
            let bytes = shard.decoded_bytes();
            if let Some(reservation) = &mut shard.prepared.reservation {
                reservation.shrink_to(bytes);
            }
        }
        Ok((context, shard))
    }

    /// Resident Arrow buffers, including blob payloads, before group reduction.
    pub fn decoded_bytes(&self) -> usize {
        self.prepared
            .batches
            .iter()
            .map(RecordBatch::get_array_memory_size)
            .sum()
    }

    pub fn generation_count(&self) -> usize {
        self.prepared.generation_count()
    }

    /// Prepared shard progress, not commit proof. Publish only after the
    /// entire group and its WAL drains have acknowledged success.
    #[must_use]
    pub fn prepared_watermark(&self) -> Option<(Uuid, u64)> {
        self.prepared.prepared_watermark()
    }

    /// Generations in this preparation's manifest snapshot that were not read.
    /// An ordered publisher must finish this suffix before publishing a later
    /// shard. New ingestion is not included in this fixed continuation bound.
    pub fn unprepared_generation_count(&self) -> usize {
        self.manifest
            .sstables
            .iter()
            .filter(|table| !self.prepared.merged_generations.contains(&table.generation))
            .count()
    }
}

fn stale(message: &str) -> LanceError {
    LanceError::io(MaintenanceErrorCode::StalePreparation.tag(message))
}

/// Bound the extra allocations made by cross-shard reduction. The existing
/// reducer stores owned ScalarValues in a growing HashSet, plus selection
/// indices and (for partially surviving batches) new Arrow buffers. Hash-table
/// growth may keep both old and new bucket arrays live during rehashing.
fn reduction_workspace(shards: &[PreparedShardMerge], key_column: &str) -> Option<usize> {
    let per_row = 4usize
        .checked_mul(std::mem::size_of::<ScalarValue>().checked_add(16)?)?
        .checked_add(2 * std::mem::size_of::<u32>() + 64)?;
    let mut bytes = 0usize;
    for shard in shards {
        for batch in &shard.prepared.batches {
            let key = batch.column(batch.schema().index_of(key_column).ok()?);
            // Rocketkeep's rollout and generic IDs are strings. Nested key
            // ScalarValues need a different allocation bound; use the already
            // deduplicated serial preparations for those types.
            if !matches!(
                key.data_type(),
                arrow_schema::DataType::Utf8 | arrow_schema::DataType::LargeUtf8
            ) {
                return None;
            }
            bytes = bytes
                .checked_add(batch.get_array_memory_size())?
                .checked_add(key.get_array_memory_size())?
                .checked_add(batch.num_rows().checked_mul(per_row)?)?;
        }
    }
    Some(bytes)
}

impl StorageBase {
    /// Read one sealed shard for a bounded group. The caller must acquire the
    /// reservation without waiting when it already holds other preparations.
    pub async fn prepare_grouped_merge_reserved(
        &self,
        reservation: MergeReservation,
    ) -> LanceResult<Option<PreparedShardMerge>> {
        let Some((manifest_store, manifest, prepared)) =
            self.prepare_merge_reserved(reservation).await?
        else {
            return Ok(None);
        };
        // Keep the full allowance: native reader children may still be joining
        // in the caller's owned preparation scope when this future returns.
        Ok(Some(PreparedShardMerge {
            uri: self.uri().to_owned(),
            key_column: self.key_column.clone(),
            manifest_store,
            manifest,
            prepared,
            parallel_encoding: false,
        }))
    }

    /// Commit ordered shards, combining data and all included coverage in one
    /// base update when the aggregate byte cap and reduction workspace fit.
    ///
    /// The caller must hold the existing table maintenance ownership scope.
    /// This method does not claim writer epochs, release leases or retry an
    /// unknown publication result. An oversized generation is supported alone;
    /// otherwise-large groups split into consecutive byte-bounded groups.
    /// Workspace contention falls back to ordered single-shard commits of the
    /// already prepared data, without waiting.
    pub fn commit_prepared_group(
        &mut self,
        mut shards: Vec<PreparedShardMerge>,
        max_bytes: usize,
    ) -> futures::future::BoxFuture<'_, LanceResult<usize>> {
        Box::pin(async move {
            if shards.is_empty() {
                return Ok(0);
            }
            if max_bytes == 0 || shards.len() > 32 {
                return Err(LanceError::invalid_input(
                    "grouped merge requires a byte cap and at most 32 shards",
                ));
            }
            self.ensure_writable()?;
            let budget = self.merge_budget.clone().ok_or_else(|| {
                LanceError::invalid_input("grouped merge requires a shared memory budget")
            })?;
            let bytes = shards.iter().try_fold(0usize, |total, shard| {
                total
                    .checked_add(shard.decoded_bytes())
                    .ok_or_else(|| LanceError::invalid_input("grouped merge byte count overflow"))
            })?;
            let mut ids = HashSet::new();
            let parallel_encoding = shards.iter().all(|shard| shard.parallel_encoding);
            for shard in &shards {
                if shard.uri != self.uri() || shard.key_column != self.key_column {
                    return Err(LanceError::invalid_input(
                        "grouped preparation belongs to another table or key column",
                    ));
                }
                if !ids.insert(shard.manifest.shard_id) {
                    return Err(LanceError::invalid_input(
                        "grouped merge includes the same shard more than once",
                    ));
                }
                if shard.prepared.staged.is_some()
                    || shard.prepared.bulk_key_delete
                    || shard.prepared.parallel_upsert_io
                    || !shard
                        .prepared
                        .reservation
                        .as_ref()
                        .is_some_and(|r| r.belongs_to(&budget))
                {
                    return Err(LanceError::invalid_input(
                        "grouped merge requires unstaged preparations from the same budget",
                    ));
                }
            }
            let validation_started = timer_start!();
            for wave in shards.chunks(METADATA_CONCURRENCY) {
                let results = futures::future::join_all(wave.iter().map(|shard| async move {
                    let current = shard.manifest_store.refresh_latest().await?;
                    let Some(current) = current else {
                        return Err(stale("grouped shard manifest disappeared; reprepare"));
                    };
                    if current.writer_epoch != shard.manifest.writer_epoch
                        || !shard.prepared.merged_generations.iter().all(|generation| {
                            current.sstables.iter().any(|s| s.generation == *generation)
                        })
                    {
                        return Err(stale(
                            "grouped shard epoch or membership changed; reprepare",
                        ));
                    }
                    Ok(())
                }))
                .await;
                // Join every admitted read before returning the first input-order error.
                for result in results {
                    result?;
                }
            }
            observe_duration!(crate::metrics::ROLLOUT_WAL_MERGE_DURATION,
                timer_elapsed!(validation_started), "phase" => "grouped_validate");

            self.refresh_latest().await?;
            self.ensure_latest_schema().await?;
            if matches!(
                self.dataset.manifest().data_storage_format.version,
                lance_file::version::ConcreteFileVersion::V1
                    | lance_file::version::ConcreteFileVersion::V2_0
            ) {
                // Older encoders detach async children outside the CPU join
                // scope. Do not promise bounded cancellation for those files.
                return Err(LanceError::invalid_input(
                    "grouped merge requires Lance V2.1 or newer files",
                ));
            }
            let schema = Arc::new(Schema::from(self.dataset.schema()));
            let coverage = crate::rollout_append::watermarks(&self.dataset).await?;
            let mut watermarks = Vec::with_capacity(shards.len());
            for shard in &shards {
                if shard.prepared.merge_schema != schema {
                    return Err(stale("grouped base schema changed; reprepare"));
                }
                let id = shard.manifest.shard_id;
                if coverage
                    .get(&id)
                    .is_some_and(|high| shard.prepared.merged_generations.iter().any(|g| g <= high))
                {
                    return Err(stale(
                        "grouped preparation overlaps committed coverage; reprepare",
                    ));
                }
                if let Some(high) = shard.prepared.merged_generations.iter().max() {
                    watermarks.push((id, *high));
                }
            }

            if shards.len() > 1 && bytes > max_bytes {
                count!(crate::metrics::ROLLOUT_WAL_GROUP_SPLITS,
                    "reason" => "aggregate_byte_cap");
                // Preserve input order, including across group boundaries. An
                // indivisible oversized shard runs alone; smaller neighbours
                // can still share publications. All preparations remain charged.
                let mut remaining = shards.into_iter().peekable();
                let mut reclaimed = 0;
                while let Some(first) = remaining.next() {
                    let mut group_bytes = first.decoded_bytes();
                    let mut group = vec![first];
                    while let Some(next) = remaining.peek() {
                        let next_bytes = next.decoded_bytes();
                        // The complete group's checked sum above also bounds
                        // every consecutive subset, including an oversized one.
                        if group_bytes + next_bytes > max_bytes {
                            break;
                        }
                        group_bytes += next_bytes;
                        group.push(remaining.next().unwrap());
                    }
                    // Revalidate each subset against the latest committed base.
                    // An unknown outcome stops here: never run a later group.
                    reclaimed += self.commit_prepared_group(group, max_bytes).await?;
                }
                return Ok(reclaimed);
            }

            // Account for output buffers, owned keys, hash-table reallocation
            // and row indices before reduction. Never wait while holding the
            // preparations. A single shard is already duplicate-free.
            let scratch = if shards.len() > 1 {
                match reduction_workspace(&shards, &self.key_column) {
                    Some(workspace) => {
                        let reservation = budget.try_reserve(workspace);
                        if reservation.is_none() {
                            count!(crate::metrics::ROLLOUT_WAL_GROUP_SPLITS,
                                "reason" => "workspace_unavailable");
                        }
                        reservation
                    }
                    None => {
                        count!(crate::metrics::ROLLOUT_WAL_GROUP_SPLITS,
                            "reason" => "workspace_unbounded");
                        None
                    }
                }
            } else {
                None
            };
            if shards.len() > 1 && scratch.is_none() {
                // No publication has started. Keep the already-read data and commit
                // it in the same order, without extra reduction memory or rereads.
                // Unknown outcomes below still return immediately to recovery.
                let mut reclaimed = 0;
                for shard in shards {
                    reclaimed += self.commit_prepared_group(vec![shard], max_bytes).await?;
                }
                return Ok(reclaimed);
            }
            let batches = if shards.len() == 1 {
                std::mem::take(&mut shards[0].prepared.batches)
            } else {
                let ordered = shards
                    .iter_mut()
                    .enumerate()
                    .map(|(index, shard)| {
                        (index as u64, std::mem::take(&mut shard.prepared.batches))
                    })
                    .collect();
                dedupe_merge_batches(ordered, &self.key_column, schema.clone())?
            };
            if !batches.is_empty() {
                // The scratch allowance covers the reduced group's payload;
                // a single shard retains its own original reservation. Encoding
                // must finish, including native CPU children, before cancellation
                // may release that charge. Publication stays in the caller.
                let hold = scratch
                    .as_ref()
                    .unwrap_or_else(|| shards[0].prepared.reservation.as_ref().unwrap())
                    .retain();
                let snapshot = Arc::new(self.dataset.clone());
                let encoding = batches.clone();
                // Never wait for workspace while retaining decoded preparations.
                // The extra encoder is only admitted for the qualified upload
                // settings and a separate, conservative buffer allowance.
                let encoding_workspace = parallel_encoding
                    .then(|| parallel_encoding_workspace(&encoding))
                    .flatten()
                    .and_then(|bytes| budget.try_reserve(bytes));
                let encoding_started = timer_start!();
                let staged = crate::merge_write_scope::run_owned_preparation(hold, async move {
                    if let Some(workspace) = encoding_workspace {
                        // Keep both the input hold (outer scope) and the extra
                        // workspace charged until every native encoder joins,
                        // including cancellation, panic and partial failure.
                        let staged = crate::merge_write_scope::run_owned_preparation(
                            workspace.retain(),
                            encode_group_files(snapshot, encoding),
                        )
                        .await;
                        drop(workspace);
                        staged?
                    } else {
                        encode_merge_files(snapshot, encoding).await
                    }
                })
                .await??;
                observe_duration!(crate::metrics::ROLLOUT_WAL_MERGE_DURATION,
                    timer_elapsed!(encoding_started), "phase" => "grouped_encode");
                let publication_started = timer_start!();
                self.merge_prepared_batches(
                    batches,
                    schema,
                    watermarks,
                    false,
                    false,
                    Some(staged),
                )
                .await?;
                observe_duration!(crate::metrics::ROLLOUT_WAL_MERGE_DURATION,
                    timer_elapsed!(publication_started), "phase" => "grouped_publish");
                self.pinned_version = None;
            }
            // Decoded buffers have been consumed; drainage retains only metadata.
            drop(scratch);
            for shard in &mut shards {
                drop(shard.prepared.reservation.take());
            }
            let drain_started = timer_start!();
            let mut reclaimed = 0;
            let publisher = &*self;
            let mut remaining = shards.into_iter();
            loop {
                let wave: Vec<_> = remaining.by_ref().take(METADATA_CONCURRENCY).collect();
                if wave.is_empty() {
                    break;
                }
                let results = futures::future::join_all(wave.into_iter().map(|shard| async move {
                    let count = shard.generation_count();
                    let drain_store = ShardManifestStore::new(
                        publisher.dataset.object_store(None).await?,
                        &publisher.dataset.branch_location().path,
                        shard.manifest.shard_id,
                        DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
                    );
                    crate::merge_write_scope::drain_generations(
                        drain_store,
                        shard.manifest.writer_epoch,
                        shard.prepared.merged_generations,
                    )
                    .await?;
                    publisher
                        .delete_shard_generation_dirs(
                            shard.manifest.shard_id,
                            &shard.prepared.merged_paths,
                        )
                        .await?;
                    Ok::<_, LanceError>(count)
                }))
                .await;
                // Publication already covers every shard. A failed drain must join
                // its admitted siblings; later waves remain for watermark recovery.
                // Each write retains the existing fenced, observed commit scope.
                for result in results {
                    reclaimed += result?;
                }
            }
            observe_duration!(crate::metrics::ROLLOUT_WAL_MERGE_DURATION,
                timer_elapsed!(drain_started), "phase" => "grouped_drain");
            Ok(reclaimed)
        })
    }
}

fn parallel_encoding_workspace(batches: &[RecordBatch]) -> Option<usize> {
    // Larger process-wide upload buffers need independent qualification.
    let setting = |name: &str, default| match std::env::var(name) {
        Ok(value) => value.parse::<usize>().ok(),
        Err(std::env::VarError::NotPresent) => Some(default),
        Err(_) => None,
    };
    if setting("LANCE_INITIAL_UPLOAD_SIZE", 5 << 20)? > 5 << 20
        || !(1..=10).contains(&setting("LANCE_UPLOAD_CONCURRENCY", 10)?)
        || batches.iter().map(RecordBatch::num_rows).sum::<usize>() < 2
    {
        return None;
    }
    batches.iter().try_fold(128usize << 20, |bytes, batch| {
        bytes.checked_add(batch.get_array_memory_size().checked_mul(2)?)
    })
}

/// Keys are already unique across all batches. Splitting rows cannot alter
/// last-writer order, and neither immutable encoder can publish a base version.
async fn encode_group_files(
    snapshot: Arc<Dataset>,
    batches: Vec<RecordBatch>,
) -> LanceResult<StagedMergeFiles> {
    let mut left_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>() / 2;
    let mut left = Vec::new();
    let mut right = Vec::new();
    for batch in batches {
        let take = left_rows.min(batch.num_rows());
        if take > 0 {
            left.push(batch.slice(0, take));
        }
        if take < batch.num_rows() {
            right.push(batch.slice(take, batch.num_rows() - take));
        }
        left_rows -= take;
    }
    // join! waits for both results even if one encoder fails. The owning
    // preparation additionally joins Lance's native CPU tasks before release.
    let (left, right) = tokio::join!(
        encode_merge_files(snapshot.clone(), left),
        encode_merge_files(snapshot, right),
    );
    let mut left = left?;
    let right = right?;
    if left.uri != right.uri || left.schema != right.schema {
        return Err(stale("parallel encoders produced different base schemas"));
    }
    left.fragments.extend(right.fragments);
    Ok(left)
}

#[cfg(test)]
mod tests;
