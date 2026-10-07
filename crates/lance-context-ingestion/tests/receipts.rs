use std::sync::Arc;

use lance_context_ingestion::{
    Binding, Consumer, Entry, Journal, Position, ReceiptIndex, Sink, Transition,
};
use object_store::{memory::InMemory, path::Path, ObjectStoreExt};

fn journal(store: Arc<InMemory>) -> Journal {
    Journal::new(
        store,
        Path::from("receipt-test"),
        Binding {
            run: "run".into(),
            schema: "schema".into(),
            partition: 0,
        },
        1 << 20,
        1,
    )
    .unwrap()
}

fn entry(sequence: u64, receipt: &str) -> Entry {
    Entry {
        sequence,
        session: "same-session".into(),
        receipt: receipt.into(),
        input_digest: format!("digest-{sequence}"),
        transition: Transition {
            delta: vec![1],
            records: vec![2],
        },
    }
}

#[tokio::test]
async fn lookup_reconciles_partial_index_and_paged_tail_then_uses_only_indexed_metadata() {
    let store = Arc::new(InMemory::new());
    let journal = journal(store.clone());
    let index = ReceiptIndex::new(journal.clone());
    let mut writer = journal.acquire().await.unwrap();
    let first = writer
        .append(vec![entry(1, "split-0/batch-0/call-0")])
        .await
        .unwrap();
    let last = writer
        .append(vec![entry(2, "split-0/batch-0/call-1")])
        .await
        .unwrap();
    let lookup = index
        .find("receipts", "split-0/batch-0/call-1")
        .await
        .unwrap();
    assert_eq!(lookup.through, last);
    assert_eq!(lookup.receipt.unwrap().sequence, 2);
    let requested = ["split-0/batch-0/call-0", "split-0/batch-0/call-1", "absent"];
    let batch = index.find_many("receipts", &requested, 2).await.unwrap();
    assert_eq!(batch.through, last);
    assert_eq!(batch.receipts.len(), 2);
    assert_eq!(batch.receipts[requested[0]].sequence, 1);
    assert_eq!(batch.receipts[requested[1]].sequence, 2);
    assert!(index
        .find("receipts", "absent")
        .await
        .unwrap()
        .receipt
        .is_none());

    // Simulate sink success before the consumer cursor can be saved.
    let mut sink = index.sink(2).unwrap();
    sink.apply(journal.binding(), &[entry(1, "split-0/batch-0/call-0")])
        .await
        .unwrap();
    assert_eq!(
        journal.consumer_position("receipts").await.unwrap(),
        Position::default()
    );
    assert_eq!(
        index
            .find("receipts", "split-0/batch-0/call-0")
            .await
            .unwrap()
            .receipt
            .unwrap()
            .sequence,
        1
    );
    let mut consumer = Consumer::open(journal.clone(), "receipts").await.unwrap();
    while consumer.consume(&mut sink, 4, 1 << 20).await.unwrap() != 0 {}
    assert_eq!(consumer.position(), &last);
    // Fully indexed reads must not fetch old payloads. The immutable links remain.
    for position in [first, last.clone()] {
        store
            .delete(&Path::from(format!(
                "receipt-test/segments/{}.json",
                position.segment.unwrap()
            )))
            .await
            .unwrap();
    }
    for (receipt, sequence) in [("split-0/batch-0/call-0", 1), ("split-0/batch-0/call-1", 2)] {
        let lookup = index.find("receipts", receipt).await.unwrap();
        assert_eq!(lookup.through, last);
        let found = lookup.receipt.unwrap();
        assert_eq!(found.sequence, sequence);
        assert_eq!(found.session, "same-session");
        assert_eq!(found.input_digest, format!("digest-{sequence}"));
    }
    assert!(index
        .find("receipts", "absent")
        .await
        .unwrap()
        .receipt
        .is_none());
    assert!(index
        .find("wrong-cursor", "split-0/batch-0/call-0")
        .await
        .is_err());
    let indexed_batch = index.find_many("receipts", &requested, 2).await.unwrap();
    assert_eq!(indexed_batch.through, batch.through);
    assert_eq!(indexed_batch.receipts, batch.receipts);
}

#[tokio::test]
async fn reused_receipt_in_committed_tail_fails_lookup_and_does_not_advance_index() {
    let journal = journal(Arc::new(InMemory::new()));
    let index = ReceiptIndex::new(journal.clone());
    let mut writer = journal.acquire().await.unwrap();
    let first = writer
        .append(vec![entry(1, "source-receipt")])
        .await
        .unwrap();
    let mut consumer = Consumer::open(journal.clone(), "receipts").await.unwrap();
    let mut sink = index.sink(1).unwrap();
    consumer.consume(&mut sink, 1, 1 << 20).await.unwrap();
    // A faulty admission owner assigned the same source receipt twice.
    writer
        .append(vec![entry(2, "source-receipt")])
        .await
        .unwrap();
    assert!(index.find("receipts", "source-receipt").await.is_err());
    assert!(consumer.consume(&mut sink, 1, 1 << 20).await.is_err());
    assert_eq!(journal.consumer_position("receipts").await.unwrap(), first);
    // Sink retry can never overwrite the original immutable source identity.
    sink.apply(journal.binding(), &[entry(1, "source-receipt")])
        .await
        .unwrap();
}

#[tokio::test]
async fn index_rejects_wrong_binding_empty_identity_and_uncommitted_sequence() {
    let journal = journal(Arc::new(InMemory::new()));
    journal.acquire().await.unwrap();
    let index = ReceiptIndex::new(journal.clone());
    assert!(index.sink(0).is_err());
    assert!(index.find("receipts", "").await.is_err());
    assert!(index.find_many("receipts", &[], 1).await.is_err());
    assert!(index.find_many("receipts", &["one"], 0).await.is_err());
    assert!(index
        .find_many("receipts", &["one", "one"], 2)
        .await
        .is_err());
    let mut sink = index.sink(1).unwrap();
    let mut wrong = journal.binding().clone();
    wrong.partition = 1;
    assert!(sink.apply(&wrong, &[entry(1, "receipt")]).await.is_err());
    assert!(sink
        .apply(journal.binding(), &[entry(1, "")])
        .await
        .is_err());
    assert!(sink
        .apply(journal.binding(), &[entry(0, "receipt")])
        .await
        .is_err());
    // Misusing Sink directly cannot make lookup attest to an uncommitted input.
    sink.apply(journal.binding(), &[entry(1, "receipt")])
        .await
        .unwrap();
    assert!(index.find("receipts", "receipt").await.is_err());
}
