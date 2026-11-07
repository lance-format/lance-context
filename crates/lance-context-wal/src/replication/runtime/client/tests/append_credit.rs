use super::*;

#[test]
fn append_reply_limit_covers_every_valid_wire_outcome() -> Result<()> {
    let identity = Scope {
        cluster: uuid::Uuid::from_u128(u128::MAX),
        layout: [u8::MAX; 32],
        policy: [u8::MAX; 32],
        node: u64::MAX,
    };
    let replies = [
        Reply::Redirect { leader: u64::MAX },
        Reply::Unknown,
        Reply::AppendNotSubmittedDiskBudget,
        Reply::Commit(Outcome::Unknown),
        Reply::Commit(Outcome::Committed(Applied::Accepted { index: u64::MAX })),
        Reply::Commit(Outcome::Committed(Applied::Rejected {
            expected_sequence: u64::MAX,
        })),
        Reply::Commit(Outcome::Committed(Applied::Capacity)),
        Reply::Commit(Outcome::Committed(Applied::ProducerFenced)),
    ];
    for reply in replies {
        // Keep the existing generic encoding: no new wire format or reply type.
        let encoded = codec().serialize(&(&identity, &reply))?;
        assert!(encoded.len() <= MAX_APPEND_META);
        let (actual_scope, actual_reply): (Scope, Reply) = codec()
            .with_limit(MAX_APPEND_META as u64)
            .deserialize(&encoded)?;
        assert_eq!(actual_scope, identity);
        assert_eq!(codec().serialize(&(actual_scope, actual_reply))?, encoded);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retained_append_replies_share_credit_and_preserve_exact_results() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let available = c.client.budgets.data.available_permits();
    let original = batch(1);
    let mut replies = Vec::new();
    let mut first_index = None;
    // Keep every response alive across connection reuse. Generic reply credit
    // would exhaust the unchanged 64 MiB budget before all these small replies.
    let result = timeout(Duration::from_secs(5), async {
        for _ in 0..32 {
            let response = c
                .client
                .request(
                    leader,
                    Request::Append {
                        group: 10,
                        batch: original.clone(),
                    },
                )
                .await?;
            let index = accepted(&response);
            assert_eq!(index, *first_index.get_or_insert(index));
            assert!(response.body.is_empty());
            assert!(response._credit.num_permits() < 64 << 10);
            replies.push(response);
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;
    let held = replies
        .iter()
        .map(|r| r._credit.num_permits())
        .sum::<usize>();
    assert_eq!(c.client.budgets.data.available_permits(), available - held);
    drop(replies);
    assert_eq!(c.client.budgets.data.available_permits(), available);
    // Clean up the actual RF3 processes even when admission times out.
    c.shutdown().await?;
    result.context("retained append replies exhausted generic reply credit")??;
    Ok(())
}
