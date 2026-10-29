//! Opt-in grouping of prepared shards into one atomic base update.
//!
//! Input order is commit order: a later shard wins for a repeated key, just as
//! it would when the existing ordered publisher commits the shards separately.
//! Generation numbers are local to a shard and must not order different shards.
use super::*;

/// An opaque, budgeted shard preparation bound to its source table.
pub struct PreparedShardMerge {
    uri: String,
    key_column: String,
    manifest_store: ShardManifestStore,
    manifest: ShardManifest,
    prepared: PreparedMerge,
}

impl PreparedShardMerge {
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
        }))
    }

    /// Commit ordered shards, combining data and all included coverage in one
    /// base update when the aggregate byte cap and reduction workspace fit.
    ///
    /// The caller must hold the existing table maintenance ownership scope.
    /// This method does not claim writer epochs, release leases or retry an
    /// unknown publication result. An oversized generation is supported alone;
    /// otherwise-large groups and workspace contention fall back to ordered
    /// single-shard commits of the already prepared data, without waiting.
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
            }

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

            // Account for output buffers, owned keys, hash-table reallocation
            // and row indices before reduction. Never wait while holding the
            // preparations. A single shard is already duplicate-free.
            let scratch = if shards.len() > 1 && bytes <= max_bytes {
                reduction_workspace(&shards, &self.key_column)
                    .and_then(|workspace| budget.try_reserve(workspace))
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
                let staged = crate::merge_write_scope::run_owned_preparation(hold, async move {
                    encode_merge_files(snapshot, encoding).await
                })
                .await??;
                self.merge_prepared_batches(
                    batches,
                    schema,
                    watermarks,
                    false,
                    false,
                    Some(staged),
                )
                .await?;
                self.pinned_version = None;
            }
            // Decoded buffers have been consumed; drainage retains only metadata.
            drop(scratch);
            for shard in &mut shards {
                drop(shard.prepared.reservation.take());
            }
            let mut reclaimed = 0;
            for shard in shards {
                let count = shard.generation_count();
                let drain_store = ShardManifestStore::new(
                    self.dataset.object_store(None).await?,
                    &self.dataset.branch_location().path,
                    shard.manifest.shard_id,
                    DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
                );
                crate::merge_write_scope::drain_generations(
                    drain_store,
                    shard.manifest.writer_epoch,
                    shard.prepared.merged_generations,
                )
                .await?;
                self.delete_shard_generation_dirs(
                    shard.manifest.shard_id,
                    &shard.prepared.merged_paths,
                )
                .await?;
                reclaimed += count;
            }
            Ok(reclaimed)
        })
    }
}

#[cfg(test)]
mod tests;
