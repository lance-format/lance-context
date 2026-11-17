//! Maintain an ID index without turning an already committed append into an error.
use super::*;
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance_index::{optimize::OptimizeOptions, scalar::ScalarIndexParams, IndexType};

const ID_INDEX: &str = "document_id_btree";

impl DocumentStore {
    /// Build or extend the ID index. Useful for stores from earlier releases and
    /// for retrying failed automatic maintenance. A prepared snapshot is a no-op;
    /// pinned snapshots may read their existing index but cannot build one.
    pub async fn prepare_point_reads(&mut self) -> Result<()> {
        let indices = self.dataset.load_indices().await?;
        let count = indices
            .iter()
            .filter(|index| index.name == ID_INDEX)
            .count();
        if count > 0 && self.dataset.unindexed_fragments(ID_INDEX).await?.is_empty() {
            return Ok(());
        }
        if self.pinned {
            return Err(Error::invalid_input(
                "cannot index a pinned document snapshot",
            ));
        }
        if self.dataset.count_rows(None).await? == 0 {
            return Ok(());
        }
        if count == 0 {
            self.dataset
                .create_index_builder(&[ID], IndexType::BTree, &ScalarIndexParams::default())
                .name(ID_INDEX.to_owned())
                .replace(true)
                .await?;
        } else {
            // Append small deltas, periodically consolidate to bound query fanout.
            let options = if count >= 8 {
                OptimizeOptions::merge(count)
            } else {
                OptimizeOptions::append()
            };
            self.dataset
                .optimize_indices(&options.index_names(vec![ID_INDEX.to_owned()]))
                .await?;
        }
        Ok(())
    }

    pub(super) async fn maintain_point_reads(&mut self) {
        // Data is already durable. Returning an append error here could lead a
        // caller to retry and duplicate it. Reads remain correct via tail scans.
        if let Err(error) = self.prepare_point_reads().await {
            tracing::warn!(%error, "document data committed but ID index maintenance failed; retry prepare_point_reads");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn id_index_covers_appends_reopen_and_backfill() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("ids.lance");
        let uri = uri.to_str().unwrap();
        let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
            .await
            .unwrap();
        let mut ids = Vec::new();
        for batch in 0..10 {
            ids.extend(store.add(vec![json!({"value": batch})]).await.unwrap());
            assert!(store
                .dataset
                .unindexed_fragments(ID_INDEX)
                .await
                .unwrap()
                .is_empty());
        }
        assert!(
            store
                .dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .filter(|index| index.name == ID_INDEX)
                .count()
                <= 8
        );
        let mut scanner = store.dataset.scan();
        scanner.filter_expr(col(ID).eq(lit(ids[0].as_str())));
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndex") || plan.contains("IndexLookup"),
            "{plan}"
        );
        let version = store.version();
        store.prepare_point_reads().await.unwrap();
        assert_eq!(store.version(), version);
        store.backfill().await.unwrap();
        assert!(store
            .dataset
            .unindexed_fragments(ID_INDEX)
            .await
            .unwrap()
            .is_empty());
        let mut reopened = DocumentStore::open(uri, None).await.unwrap();
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(reopened.get(id).await.unwrap(), Some(json!({"value": i})));
        }
        assert_eq!(reopened.get("missing").await.unwrap(), None);
        reopened.checkout(version).await.unwrap();
        reopened.prepare_point_reads().await.unwrap();
        assert_eq!(reopened.version(), version);
    }

    #[tokio::test]
    async fn legacy_stores_can_prepare_point_reads_and_pinned_unindexed_reads_stay_valid() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("legacy-ids.lance");
        let uri = uri.to_str().unwrap();
        let mut store = DocumentStore::create(uri, ShredPolicy::default(), None)
            .await
            .unwrap();
        let id = store.add(vec![json!({"value": "kept"})]).await.unwrap()[0].clone();
        store.dataset.drop_index(ID_INDEX).await.unwrap();
        let unindexed_version = store.version();
        store.checkout(unindexed_version).await.unwrap();
        assert_eq!(
            store.get(&id).await.unwrap(),
            Some(json!({"value": "kept"}))
        );
        assert!(store.prepare_point_reads().await.is_err());
        store.refresh().await.unwrap();
        store.prepare_point_reads().await.unwrap();
        assert!(store
            .dataset
            .unindexed_fragments(ID_INDEX)
            .await
            .unwrap()
            .is_empty());
    }
}
