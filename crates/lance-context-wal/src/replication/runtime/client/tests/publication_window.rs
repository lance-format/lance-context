use super::*;
use crate::{
    ingest_contract::{IngestContract, WritePolicy},
    replication::publication as p,
};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn maximum_publication_plan_round_trips_over_control_rpc() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let table = Uuid::new_v4();
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Acquire(p::Acquire {
            table,
            destination: p::Destination {
                namespace: Uuid::new_v4(),
                dataset: "table".into(),
            },
            ingest: IngestContract {
                tenant: "test".into(),
                qualified_table: "table".into(),
                schema: [7; 32],
                policy: WritePolicy::ImmutableUniqueIds,
                max_record_bytes: 4096,
                max_rows: 1,
            },
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: 0,
        }))))
        .await?;
    let Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Acquired(authority)))) =
        &response.reply
    else {
        anyhow::bail!("publisher acquisition failed");
    };
    let authority = authority.clone();
    drop(response);
    // The catalog stores the selection; this does not establish a source pin.
    let producer = Uuid::new_v4();
    let records = (1..=p::MAX_INPUTS as u64)
        .map(|index| {
            (
                index,
                Batch::new(table, 1, producer, index, Bytes::from_static(b"row")).meta,
            )
        })
        .collect();
    let plan = p::Plan {
        transaction: Uuid::new_v4(),
        read_version: 1,
        sources: BTreeMap::from([(
            20,
            p::SourceRange {
                after: 0,
                through: p::MAX_INPUTS as u64,
                records,
            },
        )]),
        next_group: 10,
    };
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Plan(p::PlanRequest {
            authority,
            plan: plan.clone(),
        }))))
        .await?;
    assert!(matches!(
        &response.reply,
        Reply::Commit(Outcome::Committed(Applied::Publisher(
            p::Reply::Planned { .. }
        )))
    ));
    drop(response);
    let response = c
        .client
        .request_routed(Request::Query {
            group: 10,
            query: query::Query::Publisher { table },
        })
        .await?;
    let Reply::Query(query::Outcome::Observed {
        value: query::Value::Publisher(Some(publisher)),
        ..
    }) = &response.reply
    else {
        anyhow::bail!("publisher query failed");
    };
    assert_eq!(publisher.progress.pending.as_ref().unwrap().plan, plan);
    drop(response);
    assert_eq!(
        c.client.budgets.control.available_permits(),
        c.client.options.control_bytes
    );
    c.shutdown().await
}
