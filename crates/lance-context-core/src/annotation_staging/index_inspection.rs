//! Explicit diagnostics of native query-visible indices on one protected base.
use super::*;
use lance::index::DatasetIndexExt;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct IndexInspection {
    pub source: Scope,
    pub version: u64,
    pub materialized: u64,
    pub fragments: usize,
    /// Query-visible native indices, grouped by name, keyed columns and type.
    /// Unsupported/hidden indices excluded by Lance are not an inventory here.
    pub indices: Vec<NativeIndexCoverage>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct IndexedColumn {
    pub field_id: i32,
    /// Schema path, not an inferred column name. None means a missing field.
    pub path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct NativeIndexCoverage {
    pub name: String,
    /// Only keyed columns; carried columns do not provide lookup acceleration.
    pub columns: Vec<IndexedColumn>,
    /// Exact native type identifier. None means unavailable, not no index.
    pub type_url: Option<String>,
    pub segments: usize,
    /// Union of segment coverage intersected with this manifest's fragments.
    /// None means at least one segment has unknown coverage.
    pub covered_fragments: Option<usize>,
    pub uncovered_fragments: Option<usize>,
}

impl AnnotationStaging {
    /// Inspect the installed base, which may lag the catalog-selected version.
    /// Uses one existing reader permit and pins local GC across metadata I/O.
    /// Version and materialized cut are captured together; subsequent writes
    /// and publication may proceed while Lance loads index metadata.
    ///
    /// This explicitly permits native index/manifest metadata I/O, including
    /// Lance's fragment-reuse coverage derivation. It never scans body rows or
    /// constructs indices. Coverage does not prove planner use or selectivity.
    pub async fn inspect_installed_indices(&self) -> Result<IndexInspection> {
        let _reader = reserve(&self.readers, 1)?;
        let (base, materialized) = {
            let state = self.state.read().await;
            (state.base.clone(), state.materialized)
        };
        let indices = base.load_indices().await?;
        Ok(IndexInspection {
            source: self.scope.clone(),
            version: base.version_id(),
            materialized,
            fragments: base.manifest().fragments.len(),
            indices: summarize(&base, &indices)?,
        })
    }
}

fn summarize(base: &Dataset, indices: &[IndexMetadata]) -> Result<Vec<NativeIndexCoverage>> {
    let current = base
        .manifest()
        .fragments
        .iter()
        .map(|fragment| {
            u32::try_from(fragment.id)
                .map_err(|_| invalid("fragment ID exceeds native index coverage range"))
        })
        .collect::<Result<RoaringBitmap>>()?;
    // Same-name segments may overlap: adding their counts would overstate
    // coverage. An absent bitmap must not turn into a reported complete index.
    let mut groups = BTreeMap::new();
    for index in indices {
        let key = (
            index.name.clone(),
            index.keyed_fields().to_vec(),
            index.index_details.as_ref().map(|d| d.type_url.clone()),
        );
        let (segments, coverage) = groups
            .entry(key)
            .or_insert((0usize, Some(RoaringBitmap::new())));
        *segments += 1;
        match (coverage.as_mut(), index.fragment_bitmap.as_ref()) {
            (Some(covered), Some(bitmap)) => *covered |= bitmap & &current,
            (_, None) => *coverage = None,
            (None, Some(_)) => {}
        }
    }
    Ok(groups
        .into_iter()
        .map(|((name, fields, type_url), (segments, coverage))| {
            let covered_fragments = coverage.map(|bitmap| bitmap.len() as usize);
            NativeIndexCoverage {
                name,
                columns: fields
                    .into_iter()
                    .map(|field_id| IndexedColumn {
                        field_id,
                        path: base.schema().field_path_minimal(field_id).ok(),
                    })
                    .collect(),
                type_url,
                segments,
                covered_fragments,
                uncovered_fragments: covered_fragments
                    .map(|covered| base.manifest().fragments.len() - covered),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatchIterator;
    use lance::dataset::{WriteMode, WriteParams};
    use lance_index::{scalar::ScalarIndexParams, IndexType};

    #[tokio::test]
    async fn native_index_inspection_preserves_partial_unknown_and_overlapping_coverage(
    ) -> Result<()> {
        let root = tempfile::tempdir()?;
        let batch = RecordBatch::try_from_iter([(
            "id",
            Arc::new(StringArray::from(vec!["a", "b"])) as arrow_array::ArrayRef,
        )])?;
        let rows = || RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let mut base = Dataset::write(rows(), root.path().to_str().unwrap(), None).await?;
        base.create_index(
            &["id"],
            IndexType::BTree,
            Some("body_id".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await?;
        let base = Arc::new(
            Dataset::write(
                rows(),
                root.path().to_str().unwrap(),
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await?,
        );
        let staging = AnnotationStaging::new_immutable_body(
            base.clone(),
            Scope {
                table_instance: "body".into(),
                source_group: 10,
                shard: 0,
            },
            7,
            Limits::default(),
        )?;
        // A metadata-only inspection must still work without body data files.
        for fragment in base.manifest().fragments.iter() {
            for file in &fragment.files {
                std::fs::remove_file(root.path().join("data").join(&file.path))?;
            }
        }
        let inspected = staging.inspect_installed_indices().await?;
        assert_eq!(inspected.version, base.version_id());
        assert_eq!(inspected.materialized, 7);
        assert_eq!(inspected.fragments, 2);
        assert_eq!(inspected.indices.len(), 1);
        let index = &inspected.indices[0];
        assert_eq!(index.columns[0].path.as_deref(), Some("id"));
        assert!(index
            .type_url
            .as_ref()
            .unwrap()
            .ends_with("BTreeIndexDetails"));
        assert_eq!(index.covered_fragments, Some(1));
        assert_eq!(index.uncovered_fragments, Some(1));
        assert_eq!(staging.observation().await.readers, 0);

        let mut segments = base.load_indices().await?.as_ref().clone();
        segments.push(segments[0].clone());
        segments[1].fragment_bitmap.as_mut().unwrap().insert(999);
        let merged = summarize(&base, &segments)?;
        assert_eq!(merged[0].segments, 2);
        assert_eq!(merged[0].covered_fragments, Some(1));
        segments[1].fragment_bitmap = None;
        let unknown = summarize(&base, &segments)?;
        assert_eq!(unknown[0].covered_fragments, None);
        assert_eq!(unknown[0].uncovered_fragments, None);
        assert!(summarize(&base, &[])?.is_empty());
        Ok(())
    }
}
