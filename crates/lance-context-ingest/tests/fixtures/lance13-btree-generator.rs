//! Write an immutable compatibility fixture with the real Lance 13 builder.
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use lance::{dataset::WriteParams, index::DatasetIndexExt, Dataset};
use lance_index::{scalar::ScalarIndexParams, IndexType};
use std::{path::PathBuf, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or("fixture path required")?);
    if path.exists() {
        return Err("fixture target already exists; never overwrite it".into());
    }
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E", "F"]))
                as arrow_array::ArrayRef,
        ),
        (
            "status",
            Arc::new(StringArray::from(vec![
                Some("ready"),
                None,
                Some("pending"),
                None,
                Some("ready"),
                Some("pending"),
            ])) as arrow_array::ArrayRef,
        ),
    ])?;
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        path.to_str().ok_or("non-UTF8 fixture path")?,
        Some(WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await?;
    assert_eq!(dataset.fragments().len(), 3);
    for (column, name) in [
        ("id", "lance_context_turn_id"),
        ("status", "lance_context_annotation_status"),
    ] {
        dataset
            .create_index_builder(&[column], IndexType::BTree, &ScalarIndexParams::default())
            .name(name.into())
            .fragments(vec![0])
            .await?;
    }
    let indices = dataset.load_indices().await?;
    assert_eq!(indices.len(), 2);
    assert!(indices.iter().all(|index| index.index_version == 0));
    let report = serde_json::json!({"lance": "13.0.0", "version": dataset.version_id(),
        "rows": dataset.count_rows(None).await?, "fragments": dataset.fragments().len(),
        "indices": indices.iter().map(|index| serde_json::json!({
            "name": index.name, "uuid": index.uuid, "version": index.index_version,
            "coverage": index.fragment_bitmap.as_ref().unwrap().iter().collect::<Vec<_>>()
        })).collect::<Vec<_>>()});
    std::fs::write(
        path.with_extension("fixture.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
