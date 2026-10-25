use super::*;

fn contract() -> IngestContract {
    IngestContract {
        tenant: "t".into(),
        qualified_table: "q".into(),
        schema: [1; 32],
        policy: WritePolicy::ImmutableUniqueIds,
        max_record_bytes: 1024,
        max_rows: 1,
    }
}

#[test]
fn binary_contract_fixture_preserves_field_order_policy_and_limits() {
    // Independent bincode v1 fixed-width layout: u64 UTF-8 byte lengths, schema
    // bytes, u32 enum discriminant, u32 byte limit and u32 row limit.
    let fixture = [
        1, 0, 0, 0, 0, 0, 0, 0, b't', 1, 0, 0, 0, 0, 0, 0, 0, b'q', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 4, 0, 0,
        1, 0, 0, 0,
    ];
    assert_eq!(bincode::serialize(&contract()).unwrap(), fixture);
    assert_eq!(
        bincode::deserialize::<IngestContract>(&fixture).unwrap(),
        contract()
    );
    let mut foreign_policy = fixture;
    foreign_policy[50] = 2;
    assert!(bincode::deserialize::<IngestContract>(&foreign_policy).is_err());
    let mut invalid_limit = fixture;
    invalid_limit[54..58].fill(0);
    assert_eq!(
        bincode::deserialize::<IngestContract>(&invalid_limit)
            .unwrap()
            .validate(),
        Err(ValidationError::InvalidRecordLimit)
    );
}

#[test]
fn annotation_binding_has_a_distinct_policy_without_changing_append_bytes() {
    let mut annotation = contract();
    annotation.policy = WritePolicy::TurnAnnotations;
    annotation.validate().unwrap();
    let mut expected = bincode::serialize(&contract()).unwrap();
    expected[50..54].copy_from_slice(&1u32.to_le_bytes());
    assert_eq!(bincode::serialize(&annotation).unwrap(), expected);
    assert_eq!(
        bincode::deserialize::<IngestContract>(&expected).unwrap(),
        annotation
    );
    assert_ne!(annotation, contract());
}

#[test]
fn schema_identity_has_an_independent_domain_and_length_fixture() {
    // SHA-256 of the documented domain, 8-byte LE length 2, and "{}", checked
    // independently with OpenSSL. This does not imply JSON canonicalization.
    assert_eq!(
        schema_fingerprint(b"{}"),
        [
            5, 36, 127, 73, 165, 116, 54, 235, 119, 118, 209, 98, 217, 13, 2, 25, 226, 70, 105,
            188, 240, 227, 113, 53, 118, 28, 0, 33, 28, 126, 136, 241,
        ]
    );
    assert_ne!(schema_fingerprint(b"{}"), schema_fingerprint(b"{ }"));
    assert_ne!(
        schema_fingerprint(b"{}"),
        <[u8; 32]>::from(Sha256::digest(b"{}"))
    );
}

#[test]
fn validation_uses_utf8_byte_limits_and_rejects_invalid_decoded_metadata() {
    let mut value = contract();
    value.tenant = "é".repeat(512);
    value.qualified_table = "表".repeat(341) + "x";
    value.max_record_bytes = MAX_RECORD_BYTES;
    value.max_rows = MAX_ROWS;
    value.validate().unwrap();
    value.tenant.push('x');
    assert_eq!(value.validate(), Err(ValidationError::TenantTooLong));
    value.tenant.clear();
    value.validate().unwrap();
    value.qualified_table.push('x');
    assert_eq!(value.validate(), Err(ValidationError::InvalidTable));
    value.qualified_table.clear();
    assert_eq!(value.validate(), Err(ValidationError::InvalidTable));
    value = contract();
    value.schema = [0; 32];
    assert_eq!(value.validate(), Err(ValidationError::MissingSchema));
    value = contract();
    for bytes in [0, MAX_RECORD_BYTES + 1] {
        value.max_record_bytes = bytes;
        assert_eq!(value.validate(), Err(ValidationError::InvalidRecordLimit));
    }
    value = contract();
    for rows in [0, MAX_ROWS + 1] {
        value.max_rows = rows;
        assert_eq!(value.validate(), Err(ValidationError::InvalidRowLimit));
    }
}
