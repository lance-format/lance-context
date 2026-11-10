use super::*;
use serde_json::json;

fn round_trip(layout: &ShredLayout, document: Value) {
    let encoded = layout.shred(document.clone()).unwrap();
    assert_eq!(layout.reconstruct(encoded).unwrap(), document);
}

#[test]
fn discovers_large_rare_text_without_declared_paths() {
    let large = "unique document text ".repeat(4096);
    let documents = vec![
        json!({"body": large, "kind": "a"}),
        json!({"kind": "b"}),
        json!({"kind": "c"}),
    ];
    let policy = ShredPolicy {
        max_columns: 1,
        ..Default::default()
    };
    let layout = ShredLayout::default().evolve(&documents, &policy).unwrap();
    assert_eq!(
        layout.fields,
        vec![ShredField {
            path: vec!["body".into()],
            data_type: ShredType::String
        }]
    );
    let encoded = layout.shred(documents[0].clone()).unwrap();
    assert_eq!(encoded.residual, json!({"kind": "a"}));
    assert_eq!(encoded.values, vec![Some(json!(large))]);
    assert_eq!(layout.reconstruct(encoded).unwrap(), documents[0]);
}

#[test]
fn preserves_missing_null_empty_and_mixed_types() {
    let layout = ShredLayout::default()
        .evolve(&[json!({"x": "text"})], &ShredPolicy::default())
        .unwrap();
    for document in [
        json!({}),
        json!({"x": null}),
        json!({"x": ""}),
        json!({"x": false}),
        json!({"x": []}),
        json!({"x": {}}),
        json!({"x": 42}),
        Value::Null,
        json!([{"x": "a"}]),
    ] {
        round_trip(&layout, document);
    }
    assert_eq!(
        layout.shred(json!({"x": null})).unwrap().residual,
        json!({"x": null})
    );
    assert_eq!(layout.shred(json!({"x": ""})).unwrap().residual, json!({}));
}

#[test]
fn literal_keys_nested_objects_arrays_and_numbers_round_trip() {
    let document = json!({"": {"a/b": {"~.x": "text"}}, "array": [{"text": "keep"}, null], "signed": i64::MIN, "unsigned": u64::MAX, "float": 1.25, "integral_float": 1.0, "bool": true});
    let layout = ShredLayout::default()
        .evolve(std::slice::from_ref(&document), &ShredPolicy::default())
        .unwrap();
    assert_eq!(layout.fields.len(), 6);
    let encoded = layout.shred(document.clone()).unwrap();
    assert_eq!(
        encoded.residual,
        json!({"": {"a/b": {}}, "array": [{"text": "keep"}, null]})
    );
    assert_eq!(layout.reconstruct(encoded).unwrap(), document);
}

#[test]
fn evolution_is_additive_deterministic_and_bounded() {
    let policy = ShredPolicy {
        max_columns: 2,
        ..Default::default()
    };
    let initial = ShredLayout::default()
        .evolve(&[json!({"first": "a"})], &policy)
        .unwrap();
    let documents = [json!({"later": "longer", "third": true, "first": 9})];
    let evolved = initial.evolve(&documents, &policy).unwrap();
    assert_eq!(evolved.fields[0], initial.fields[0]);
    assert_eq!(evolved.fields[1].path, vec!["later"]);
    assert_eq!(evolved, initial.evolve(&documents, &policy).unwrap());
    assert_eq!(
        evolved,
        evolved
            .evolve(&[json!({"another": "huge".repeat(10000)})], &policy)
            .unwrap()
    );
    round_trip(&evolved, json!({"first": "a"}));
    round_trip(&evolved, documents[0].clone());
}

#[test]
fn overlapping_paths_remain_lossless_across_shape_changes() {
    let policy = ShredPolicy::default();
    let initial = ShredLayout::default()
        .evolve(&[json!({"a": "text"})], &policy)
        .unwrap();
    let evolved = initial.evolve(&[json!({"a": {"b": 1}})], &policy).unwrap();
    assert_eq!(evolved.fields.len(), 2);
    round_trip(&evolved, json!({"a": "text"}));
    round_trip(&evolved, json!({"a": {"b": 1}}));
    round_trip(&evolved, json!({"a": null}));
}

#[test]
fn profiling_budgets_leave_unselected_values_in_residual() {
    let document = json!({"a": 1, "b": {"c": "nested"}, "long_key": "value"});
    let policies = [
        ShredPolicy {
            sample_nodes: 1,
            ..Default::default()
        },
        ShredPolicy {
            max_candidates: 1,
            ..Default::default()
        },
        ShredPolicy {
            max_depth: 1,
            ..Default::default()
        },
        ShredPolicy {
            max_path_bytes: 1,
            ..Default::default()
        },
    ];
    for policy in policies {
        let layout = ShredLayout::default()
            .evolve(std::slice::from_ref(&document), &policy)
            .unwrap();
        assert!(layout.fields.len() < 3);
        round_trip(&layout, document.clone());
    }
    let policy = ShredPolicy {
        sample_rows: 1,
        ..Default::default()
    };
    let layout = ShredLayout::default()
        .evolve(&[json!({"first": 1}), json!({"second": 2})], &policy)
        .unwrap();
    assert_eq!(layout.fields.len(), 1);
    assert_eq!(layout.fields[0].path, vec!["first"]);
}

#[test]
fn serialized_layout_and_policy_keep_their_meaning() {
    let policy = ShredPolicy::default();
    let layout = ShredLayout::default()
        .evolve(&[json!({"text": "value"})], &policy)
        .unwrap();
    let reopened: ShredLayout =
        serde_json::from_str(&serde_json::to_string(&layout).unwrap()).unwrap();
    assert_eq!(layout, reopened);
    assert_eq!(
        policy,
        serde_json::from_str::<ShredPolicy>(&serde_json::to_string(&policy).unwrap()).unwrap()
    );
    round_trip(&reopened, json!({"text": "different"}));
}

#[test]
fn rejects_corrupt_layouts_and_encoded_values() {
    let layout = ShredLayout::default()
        .evolve(&[json!({"a": "text"})], &ShredPolicy::default())
        .unwrap();
    let mut duplicate = layout.clone();
    duplicate.fields.push(duplicate.fields[0].clone());
    assert!(duplicate.validate().is_err());
    assert!(ShredLayout {
        fields: vec![ShredField {
            path: vec![],
            data_type: ShredType::String
        }]
    }
    .validate()
    .is_err());
    for encoded in [
        ShreddedDocument {
            residual: json!({}),
            values: vec![],
        },
        ShreddedDocument {
            residual: json!({}),
            values: vec![Some(json!(1))],
        },
        ShreddedDocument {
            residual: json!({"a": null}),
            values: vec![Some(json!("text"))],
        },
        ShreddedDocument {
            residual: json!([]),
            values: vec![Some(json!("text"))],
        },
    ] {
        assert!(layout.reconstruct(encoded).is_err());
    }
    assert!(ShredPolicy {
        min_frequency: f64::NAN,
        ..Default::default()
    }
    .validate()
    .is_err());
    assert!(ShredPolicy {
        max_depth: 129,
        ..Default::default()
    }
    .validate()
    .is_err());
}
