//! Network-backed catalog for the real Lance manifest handler.
use super::*;
use crate::replication::publication::{lance::ManifestCatalog, ReserveManifest};

/// Binds ManagedManifest to the same physical-node pools used by ingestion.
/// Resolving/authorizing the table's storage namespace remains the caller's job.
pub struct Catalog {
    client: Arc<Client>,
}
impl std::fmt::Debug for Catalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalCatalog")
            .field("cluster", &self.client.cluster)
            .field("group", &self.client.catalog)
            .finish_non_exhaustive()
    }
}
impl Client {
    pub fn manifest_catalog(self: &Arc<Self>) -> Catalog {
        Catalog {
            client: self.clone(),
        }
    }
}
#[async_trait::async_trait]
impl ManifestCatalog for Catalog {
    async fn read_manifest(
        &self,
        table: uuid::Uuid,
        version: Option<u64>,
    ) -> Result<query::Outcome> {
        let response = self
            .client
            .request_routed(Request::Query {
                group: self.client.catalog,
                query: query::Query::Manifest { table, version },
            })
            .await?;
        match &response.reply {
            Reply::Query(value) => Ok(value.clone()),
            _ => anyhow::bail!("manifest query outcome unknown"),
        }
    }
    async fn reserve_manifest(&self, request: ReserveManifest) -> Result<Outcome> {
        let response = self
            .client
            .request_routed(Request::Commit(Box::new(Command::Reserve(request))))
            .await?;
        match &response.reply {
            Reply::Commit(value) => Ok(value.clone()),
            _ => anyhow::bail!("manifest reservation outcome unknown"),
        }
    }
}
