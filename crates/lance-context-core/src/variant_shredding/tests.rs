use super::*;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use futures::TryStreamExt;
use parquet_variant_compute::VariantType;
use serde_json::json;

fn input(documents: &[Option<Value>]) -> VariantArray {
    json_values_to_variant(documents.iter().map(Option::as_ref)).unwrap()
}

fn decoded(array: &VariantArray) -> Vec<Option<Value>> {
    variant_to_json_values(array).unwrap()
}

#[test]
fn official_kernels_round_trip_null_missing_shapes_and_numeric_boundaries() {
    let selected = vec![json!({"body": "large text", "count": 1, "nested": {"flag": true}})];
    let layout = AutoVariantLayout::infer(&selected, &ShredPolicy::default()).unwrap();
    let documents = vec![
        Some(selected[0].clone()),
        None,
        Some(Value::Null),
        Some(json!({})),
        Some(json!({"body": null})),
        Some(json!({"body": 7, "count": 1.5})),
        Some(json!({"body": "text", "count": true})),
        Some(json!({"body": "text", "count": i64::MIN})),
        Some(json!({"body": "text", "count": u64::MAX})),
        Some(json!({"body": "text", "extra": [null, {}, [], {"k": "v"}]})),
        Some(json!([1, "x", null])),
        Some(json!("root")),
        Some(json!({"nested": "changed"})),
    ];
    let result = layout.shred(&input(&documents)).unwrap();
    assert!(result
        .field("document")
        .has_valid_extension_type::<VariantType>());
    assert_eq!(decoded(&result), documents);
    assert_eq!(decoded(&unshred_variant(&result).unwrap()), documents);
    assert!(result.is_null(1));
    assert!(!result.is_null(2));
    let typed = result.typed_value_field().unwrap();
    assert!(typed.is_valid(0));
    for row in [10, 11] {
        assert!(
            typed.is_null(row),
            "non-object row {row} must stay residual"
        );
    }
}

#[test]
fn dominant_text_is_stored_once_in_standard_typed_children() {
    let body = "uniquepayload".repeat(100_000);
    let documents = vec![Some(json!({"body": body, "extra": ["retained"]}))];
    let layout =
        AutoVariantLayout::infer(&[documents[0].clone().unwrap()], &ShredPolicy::default())
            .unwrap();
    let result = layout.shred(&input(&documents)).unwrap();
    let typed = result
        .typed_value_field()
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let body_field = typed
        .column_by_name("body")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert!(body_field.column_by_name("value").unwrap().is_null(0));
    let text = body_field
        .column_by_name("typed_value")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(text.value(0), body);
    let residual = result
        .value_field()
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::BinaryArray>()
        .unwrap();
    assert!(residual.value(0).len() < 100);
    assert_eq!(decoded(&result), documents);
}

#[test]
fn automatic_reselection_can_replace_exhausted_and_changed_layouts() {
    let metadata: serde_json::Map<String, Value> =
        (0..32).map(|i| (format!("field{i}"), json!(i))).collect();
    let first =
        AutoVariantLayout::infer(&[Value::Object(metadata)], &ShredPolicy::default()).unwrap();
    assert_eq!(first.fields().len(), 32);
    let later = vec![json!({"body": "late text ".repeat(8192), "field0": "changed"})];
    let next = AutoVariantLayout::infer(&later, &ShredPolicy::default()).unwrap();
    assert!(next.fields().iter().any(|f| f.path == ["body"]));
    assert!(next
        .fields()
        .iter()
        .any(|f| f.path == ["field0"] && f.data_type == ShredType::String));
    let rows = later.into_iter().map(Some).collect::<Vec<_>>();
    let old = first.shred(&input(&rows)).unwrap();
    assert_eq!(decoded(&next.shred(&old).unwrap()), rows);
}

#[tokio::test]
async fn lance_persists_standard_variant_without_vendor_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("variant.lance");
    let documents = vec![
        Some(json!({"body": "hello", "nested": {"tag": "value"}})),
        Some(json!({"body": 7})),
        None,
        Some(Value::Null),
    ];
    let selected = vec![documents[0].clone().unwrap()];
    let layout = AutoVariantLayout::infer(&selected, &ShredPolicy::default()).unwrap();
    let variant = layout.shred(&input(&documents)).unwrap();
    let schema = Arc::new(Schema::new(vec![variant.field("document")]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(variant.into_inner())]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let dataset = lance::Dataset::write(reader, path.to_str().unwrap(), None)
        .await
        .unwrap();
    let schema = Schema::from(dataset.schema());
    assert!(schema.field(0).has_valid_extension_type::<VariantType>());
    let batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let actual = batches
        .iter()
        .flat_map(|batch| decoded(&VariantArray::try_new(batch.column(0).as_ref()).unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, documents);
    let mut scanner = dataset.scan();
    scanner
        .project(&["document.typed_value.body.typed_value"])
        .unwrap();
    let batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        documents.len()
    );
}

#[test]
fn accepts_upstream_json_arrays_literal_keys_and_empty_batches() {
    let documents = vec![json!({"a.b": {"": "quoted key"}, "quote\"": "kept"})];
    let layout = AutoVariantLayout::infer(&documents, &ShredPolicy::default()).unwrap();
    let strings: ArrayRef = Arc::new(StringArray::from(vec![documents[0].to_string()]));
    let upstream = parquet_variant_compute::json_to_variant(&strings).unwrap();
    assert_eq!(
        decoded(&layout.shred(&upstream).unwrap()),
        vec![Some(documents[0].clone())]
    );
    let empty = json_values_to_variant(std::iter::empty()).unwrap();
    let result = layout.shred(&empty).unwrap();
    assert_eq!(result.len(), 0);
    assert_eq!(
        result.data_type(),
        layout.shred(&upstream).unwrap().data_type()
    );
}

#[test]
fn type_drift_keeps_sibling_text_shredded_and_preserves_every_original_type() {
    let selected = json!({
        "body": "selected text", "count": 1, "ratio": 1.5, "flag": true,
        "nested": {"text": "selected nested text", "count": 1}, "a.b": {"": 1}
    });
    let layout = AutoVariantLayout::infer(&[selected], &ShredPolicy::default()).unwrap();
    let body = "unique sibling payload ".repeat(50_000);
    let drift = [
        json!(true),
        json!(1.5),
        json!("7"),
        json!(u64::MAX),
        json!(null),
        json!({"key": [1, "text", null]}),
        json!([1, false]),
        json!(-9),
    ];
    let mut documents = drift
        .iter()
        .map(|value| {
            Some(json!({
                "body": body, "count": value, "ratio": value, "flag": value,
                "nested": {"text": body, "count": value}, "a.b": {"": value}
            }))
        })
        .collect::<Vec<_>>();
    documents.extend([
        Some(json!({"body": body, "nested": "changed object"})),
        Some(json!({"body": body, "nested": null})),
        Some(json!({"body": body})),
        None,
        Some(Value::Null),
    ]);
    let result = layout.shred(&input(&documents)).unwrap();
    assert_eq!(decoded(&result), documents);
    // Re-shredding and sliced Arrow inputs must retain the same semantics.
    assert_eq!(decoded(&layout.shred(&result).unwrap()), documents);
    let sliced = result.inner().slice(1, documents.len() - 2);
    assert_eq!(
        decoded(
            &layout
                .shred(&VariantArray::try_new(&sliced).unwrap())
                .unwrap()
        ),
        documents[1..documents.len() - 1]
    );
    let typed = result
        .typed_value_field()
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let body_field = typed
        .column_by_name("body")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let text = body_field
        .column_by_name("typed_value")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for row in 0..documents.len() - 2 {
        assert!(result.value_field().unwrap().is_null(row));
        assert_eq!(text.value(row), body);
        assert!(body_field.column_by_name("value").unwrap().is_null(row));
    }
    let nested = typed
        .column_by_name("nested")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap()
        .column_by_name("typed_value")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let nested_text = nested
        .column_by_name("text")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    for row in 0..drift.len() {
        assert!(nested_text
            .column_by_name("typed_value")
            .unwrap()
            .is_valid(row));
        assert!(nested_text.column_by_name("value").unwrap().is_null(row));
    }
}

#[test]
fn malformed_external_variant_buffers_return_errors() {
    use arrow_array::BinaryArray;
    let layout = AutoVariantLayout::infer(&[json!({"count": 1})], &ShredPolicy::default()).unwrap();
    for metadata in [Some(&[0xff][..]), None] {
        let malformed = VariantArray::from_parts(
            Arc::new(BinaryArray::from(vec![metadata])),
            Some(Arc::new(BinaryArray::from(vec![&[0xff][..]]))),
            None,
            None,
        );
        assert!(layout.shred(&malformed).is_err());
    }
}
