use super::*;
use arrow_array::{new_null_array, ArrayRef, LargeBinaryArray, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use std::collections::HashMap;

fn input() -> Result<(TurnIngestionSpec, RecordBatch)> {
    let mut fields = vec![
        Field::new("session_id", DataType::Utf8, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            "lance-encoding:compression".into(),
            "zstd".into(),
        )])),
        Field::new(
            "attributes",
            DataType::Struct(vec![Field::new("name", DataType::Utf8, true)].into()),
            true,
        ),
    ];
    fields.extend((0..60).map(|i| Field::new(format!("metric_{i}"), DataType::Int64, true)));
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        HashMap::from([("projector".into(), "body-v1".into())]),
    ));
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["session", "session"])),
        Arc::new(StringArray::from(vec!["one", "two"])),
        Arc::new(LargeBinaryArray::from(vec![Some(&b"body"[..]), None])),
    ];
    columns.extend(
        schema.fields()[3..]
            .iter()
            .map(|f| new_null_array(f.data_type(), 2)),
    );
    let spec = TurnIngestionSpec::new(
        schema.clone(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    Ok((spec, RecordBatch::try_new(schema, columns)?))
}

#[test]
fn full_body_binding_preserves_columns_metadata_and_arrow_owners() -> Result<()> {
    let (spec, original) = input()?;
    let bound = batch(&spec, &Limits::default(), Shard { bucket: 7 }, &original)?;
    assert_eq!(bound.num_columns(), 64);
    assert_eq!(bound.schema().fields(), original.schema().fields());
    assert_eq!(bound.schema().metadata()["projector"], "body-v1");
    for (before, after) in original.columns().iter().zip(bound.columns()) {
        assert!(Arc::ptr_eq(before, after), "binding copied an Arrow array");
    }
    let restored = TurnIngestionSpec::from_schema(&bound.schema())?.unwrap();
    assert_eq!(restored.body_schema(), spec.body_schema());
    assert_eq!(restored.primary_key(), spec.primary_key());
    assert_eq!(restored.routing_column(), spec.routing_column());
    Ok(())
}

#[test]
fn full_body_decode_keeps_escaped_routing_values_and_body_columns() -> Result<()> {
    let (spec, original) = input()?;
    let spec = TurnIngestionSpec::new(
        spec.body_schema().clone(),
        spec.primary_key().to_vec(),
        "id",
    )?;
    let routing = ["quote\"\\雪", "plain"];
    let mut columns = original.columns().to_vec();
    columns[spec.body_schema().index_of("id")?] = Arc::new(StringArray::from(routing.to_vec()));
    let original = RecordBatch::try_new(original.schema(), columns)?;
    let wire = batch(&spec, &Limits::default(), Shard { bucket: 7 }, &original)?;
    let mut encoded = Vec::new();
    {
        let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut encoded, &wire.schema())?;
        writer.write(&wire)?;
        writer.finish()?;
    }

    let mixed::Decoded::Creates { rows, bodies } =
        decode(Bytes::from(encoded), wire.schema(), 2, 4096)?
    else {
        panic!("full-body IPC must decode as creates");
    };
    assert_eq!(rows.len(), 2);
    for (row, value) in rows.iter().zip(routing) {
        assert_eq!(row.turn, spec.business_key(&["session", value], 4096)?);
        assert_eq!(row.rollout_id, value);
    }
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0].schema(), storage_schema(&spec)?);
    assert_eq!(bodies[0].columns(), original.columns());
    Ok(())
}

#[test]
fn body_contract_binds_routing_limits_and_reserved_metadata() -> Result<()> {
    let (spec, original) = input()?;
    let binding = TableBinding {
        table: Uuid::from_u128(2),
        destination: Destination {
            namespace: Uuid::from_u128(3),
            dataset: "turns.lance".into(),
        },
        incarnation: Uuid::from_u128(4),
        tenant: "tenant".into(),
        qualified_table: "turns".into(),
        max_record_bytes: 1 << 20,
        max_rows: 128,
    };
    let limits = Limits::default();
    let shard = Shard { bucket: 7 };
    let bound = contract(&binding, &spec, &limits, shard)?;
    assert_eq!(bound.policy, WritePolicy::TurnIngestion);
    assert_eq!(bound.max_record_bytes, binding.max_record_bytes);
    assert_eq!(bound.max_rows, binding.max_rows);
    assert_eq!(bound.tenant, binding.tenant);
    assert_eq!(bound.qualified_table, binding.qualified_table);
    let encoded = batch(&spec, &limits, shard, &original)?;
    assert_eq!(bound.schema, arrow_schema_fingerprint(&encoded.schema())?);
    assert_ne!(
        bound.schema,
        contract(&binding, &spec, &limits, Shard { bucket: 8 })?.schema
    );
    let changed = Limits {
        value_bytes: limits.value_bytes / 2,
        ..limits.clone()
    };
    assert_ne!(
        bound.schema,
        contract(&binding, &spec, &changed, shard)?.schema
    );
    let routed_by_id = TurnIngestionSpec::new(
        spec.body_schema().clone(),
        spec.primary_key().to_vec(),
        "id",
    )?;
    assert_ne!(
        bound.schema,
        contract(&binding, &routed_by_id, &limits, shard)?.schema
    );
    assert!(wire_schema(&spec, &limits, Shard { bucket: 65_536 }).is_err());
    let invalid = Limits {
        readers: 0,
        ..limits.clone()
    };
    assert!(wire_schema(&spec, &invalid, shard).is_err());
    for key in [KEY, "lance-context.annotation-mutations"] {
        let mut metadata = spec.body_schema().metadata().clone();
        metadata.insert(key.into(), "1".into());
        let foreign = Arc::new(Schema::new_with_metadata(
            spec.body_schema().fields().clone(),
            metadata,
        ));
        let foreign_spec =
            TurnIngestionSpec::new(foreign.clone(), spec.primary_key().to_vec(), "session_id")?;
        assert!(wire_schema(&foreign_spec, &limits, shard).is_err());
        let foreign_batch = RecordBatch::try_new(foreign, original.columns().to_vec())?;
        assert!(batch(&spec, &limits, shard, &foreign_batch).is_err());
    }
    Ok(())
}

#[tokio::test]
async fn historical_lookup_accepts_bound_storage_and_rejects_foreign_identity() -> Result<()> {
    let (spec, original) = input()?;
    let schema = storage_schema(&spec)?;
    let stored = RecordBatch::try_new(schema.clone(), original.columns().to_vec())?;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("turns.lance");
    let base = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(stored)], schema),
            path.to_str().unwrap(),
            None,
        )
        .await?,
    );
    let key = spec.row_key(&original, 0, 4096)?;
    let missing = spec.business_key(&["session", "missing"], 4096)?;
    let budget = MergeMemoryBudget::new(1 << 20);
    let lookup = spec
        .lookup_existing(base.clone(), &[&key, &missing], &budget)
        .await?
        .unwrap();
    assert_eq!(lookup.get(&key), Some(true));
    assert_eq!(lookup.get(&missing), Some(false));
    assert!(lookup.matches(&base));
    drop(lookup);
    assert_eq!(budget.reserved(), 0);
    // Same physical columns do not authorize a different persisted routing rule.
    let foreign = TurnIngestionSpec::new(
        spec.body_schema().clone(),
        spec.primary_key().to_vec(),
        "id",
    )?;
    assert!(foreign
        .lookup_existing(base, &[&key], &budget)
        .await
        .is_err());
    assert_eq!(budget.reserved(), 0);
    Ok(())
}
