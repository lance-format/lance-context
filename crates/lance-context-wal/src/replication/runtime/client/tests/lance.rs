use super::*;
use crate::{
    ingest_contract::{arrow_schema_fingerprint, IngestContract, WritePolicy},
    replication::publication as p,
};
use ::lance::{
    dataset::{builder::DatasetBuilder, WriteParams},
    Dataset,
};
use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance_table::io::commit::{ConditionalPutCommitHandler, ManifestNamingScheme};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn actual_lance_reader_uses_tls_catalog_after_all_wal_nodes_restart() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let payloads = (0..16)
        .map(|id| format!("{id}:{}", "payload".repeat(8192)))
        .collect::<Vec<_>>();
    let input = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..16)),
            Arc::new(StringArray::from(payloads.clone())),
        ],
    )?;
    let uri = dir
        .path()
        .join("dataset.lance")
        .to_str()
        .unwrap()
        .to_owned();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(input)], schema.clone()),
        &uri,
        Some(WriteParams {
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            ..Default::default()
        }),
    )
    .await?;
    let location = dataset.manifest_location();
    let base = object_store::path::Path::parse(
        location
            .path
            .as_ref()
            .rsplit_once("/_versions/")
            .context("dataset manifest base")?
            .0,
    )?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let table = uuid::Uuid::new_v4();
    let incarnation = uuid::Uuid::new_v4();
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Acquire(p::Acquire {
            table,
            destination: p::Destination {
                namespace: uuid::Uuid::new_v4(),
                dataset: "dataset.lance".into(),
            },
            ingest: IngestContract {
                tenant: String::new(),
                qualified_table: "table".into(),
                schema: arrow_schema_fingerprint(&schema)?,
                policy: WritePolicy::ImmutableUniqueIds,
                max_record_bytes: 8 << 20,
                max_rows: 1024,
            },
            owner: uuid::Uuid::new_v4(),
            request: uuid::Uuid::new_v4(),
            expected_epoch: 0,
        }))))
        .await?;
    let Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Acquired(authority)))) =
        &response.reply
    else {
        anyhow::bail!("publisher acquisition failed: {:?}", response.reply);
    };
    let authority = authority.clone();
    drop(response);
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Bootstrap(
            p::BootstrapManifest {
                authority,
                incarnation,
                request: uuid::Uuid::new_v4(),
                version: 1,
                object: p::ManifestObject {
                    path: format!(
                        "_versions/{}",
                        location.path.filename().context("manifest filename")?
                    ),
                    bytes: location.size.context("written manifest size")?,
                    v2_names: location.naming_scheme == ManifestNamingScheme::V2,
                },
                transaction: uuid::Uuid::parse_str(
                    &dataset
                        .read_transaction_by_version(1)
                        .await?
                        .context("genesis transaction")?
                        .uuid,
                )?,
            },
        ))))
        .await?;
    ensure!(
        matches!(
            response.reply,
            Reply::Commit(Outcome::Committed(Applied::Publisher(
                p::Reply::ManifestReserved(_)
            )))
        ),
        "manifest was not durably registered"
    );
    drop(response);
    async fn read(
        client: &Arc<Client>,
        uri: &str,
        base: &object_store::path::Path,
        table: uuid::Uuid,
        incarnation: uuid::Uuid,
        expected: &[String],
    ) -> Result<()> {
        let catalog = Arc::new(client.manifest_catalog());
        let handler = p::lance::ManagedManifest::reader(
            catalog,
            base.clone(),
            &layout(),
            table,
            incarnation,
        )?;
        let loaded = DatasetBuilder::from_uri(uri)
            .with_commit_handler(Arc::new(handler))
            .load()
            .await?;
        assert_eq!(loaded.version().version, 1);
        let batches: Vec<_> = loaded.scan().try_into_stream().await?.try_collect().await?;
        let mut rows = BTreeMap::new();
        for batch in batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let values = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..batch.num_rows() {
                assert!(rows
                    .insert(ids.value(i) as usize, values.value(i).to_owned())
                    .is_none());
            }
        }
        assert_eq!(rows, expected.iter().cloned().enumerate().collect());
        Ok(())
    }
    read(&c.client, &uri, &base, table, incarnation, &payloads).await?;
    c.shutdown().await?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    read(&c.client, &uri, &base, table, incarnation, &payloads).await?;
    c.shutdown().await
}
