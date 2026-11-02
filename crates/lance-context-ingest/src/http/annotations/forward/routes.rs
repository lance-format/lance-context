//! Bounded control messages select a configured owner. Arrow updates remain
//! opaque streams, routed by their existing authenticated producer credential.
use super::*;
use crate::annotations::Shard;

struct Route {
    ingress: Ingress,
    address: reqwest::Url,
    scope: [u8; 32],
}

pub(super) struct Routes {
    explicit: bool,
    owners: BTreeMap<u32, Route>,
}

impl Routes {
    pub(super) fn new(ingress: Ingress, address: reqwest::Url) -> Result<Self> {
        let explicit = Shard::from_ingress(&ingress)?.is_some();
        let mut routes = Self {
            explicit,
            owners: BTreeMap::new(),
        };
        routes.insert(ingress, address)?;
        Ok(routes)
    }

    pub(super) fn insert(&mut self, ingress: Ingress, address: reqwest::Url) -> Result<()> {
        let shard = Shard::from_ingress(&ingress)?;
        ensure!(
            shard.is_some() == self.explicit,
            "mixed legacy and explicit annotation routes"
        );
        let bucket = shard.map_or(0, |s| s.bucket);
        if let Some(first) = self.owners.values().next() {
            ensure!(self.explicit, "duplicate annotation route");
            let a = &first.ingress;
            let mut contract = ingress.scope.contract.clone();
            contract.schema = a.scope.contract.schema;
            ensure!(
                ingress.layout.buckets.len() == a.layout.buckets.len()
                    && contract == a.scope.contract
                    && Shard::same_wire_family(&ingress.schema, &a.schema)
                    && Arc::ptr_eq(&ingress.budget, &a.budget),
                "annotation routes require the same bucket count, wire contract and shared ingress budget"
            );
            // Key routing depends on the logical contract and bucket count,
            // not cluster placement. Keep each cluster's immutable layout and
            // every producer's complete scope; this does not migrate a shard.
            ensure!(
                self.owners.values().all(|route| {
                    route.ingress.scope.cluster != ingress.scope.cluster
                        || route.ingress.scope.layout == ingress.scope.layout
                }),
                "conflicting layouts within annotation cluster"
            );
            ensure!(
                self.owners
                    .values()
                    .all(|r| r.ingress.scope.table != ingress.scope.table
                        && r.ingress.scope.destination != ingress.scope.destination
                        && r.address != address),
                "duplicate physical annotation route"
            );
        }
        let scope = receipt::scope_hash(&ingress.scope)?;
        ensure!(
            !self.owners.contains_key(&bucket),
            "duplicate annotation bucket route"
        );
        self.owners.insert(
            bucket,
            Route {
                ingress,
                address,
                scope,
            },
        );
        Ok(())
    }

    pub(super) fn validate(&self) -> Result<()> {
        let first = self
            .owners
            .values()
            .next()
            .context("empty annotation routes")?;
        ensure!(
            self.owners.len() == first.ingress.layout.buckets.len(),
            "incomplete annotation bucket routes"
        );
        Ok(())
    }

    fn credential(&self, keys: &ReceiptKeys, token: &str) -> HttpResult<u32> {
        let (scope, bucket) = keys
            .producer_route(token)
            .map_err(|_| Failure::bad("invalid_producer"))?;
        if !self
            .owners
            .get(&bucket)
            .is_some_and(|route| route.scope == scope)
        {
            return Err(Failure::bad("foreign_producer_route"));
        }
        Ok(bucket)
    }

    pub(super) async fn select(
        &self,
        request: Request,
        operation: Operation,
        length: usize,
        keys: &ReceiptKeys,
        options: &Options,
    ) -> HttpResult<(Request, reqwest::Url)> {
        if !self.explicit {
            return Ok((request, self.owners[&0].address.clone()));
        }
        let (request, bucket) = if matches!(operation, Operation::Update) {
            if request.headers().get_all(PRODUCER).iter().count() != 1 {
                return Err(Failure::bad("ambiguous_producer_headers"));
            }
            let token = request
                .headers()
                .get(PRODUCER)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| Failure::bad("missing_producer"))?;
            let bucket = self.credential(keys, token)?;
            (request, bucket)
        } else {
            // Read only the bounded JSON control body, retain its original bytes
            // and headers, and never reserialize it to forward the request.
            let (parts, input) = request.into_parts();
            let bytes = body::read_exact(Request::new(input), length, options).await?;
            let bucket = match operation {
                Operation::Open => {
                    serde_json::from_slice::<ProducerOpen>(&bytes)
                        .map_err(|_| Failure::bad("invalid_open"))?
                        .bucket
                }
                Operation::Read => {
                    let point: PointRequest = serde_json::from_slice(&bytes)
                        .map_err(|_| Failure::bad("invalid_point_read"))?;
                    if point.turn.is_empty() {
                        return Err(Failure::bad("invalid_point_route"));
                    }
                    let ingress = &self.owners.values().next().unwrap().ingress;
                    ingress
                        .layout
                        .session_bucket(
                            &ingress.scope.contract.tenant,
                            &ingress.scope.contract.qualified_table,
                            &point.turn,
                        )
                        .map_err(|_| Failure::bad("invalid_point_route"))?
                }
                Operation::Close => {
                    let close: CloseRequest = serde_json::from_slice(&bytes)
                        .map_err(|_| Failure::bad("invalid_close"))?;
                    self.credential(keys, &close.producer)?
                }
                Operation::FilterStatus => {
                    serde_json::from_slice::<StatusRequest>(&bytes)
                        .map_err(|_| Failure::bad("invalid_status_filter"))?
                        .bucket
                }
                Operation::Update => unreachable!(),
            };
            (
                Request::from_parts(parts, axum::body::Body::from(bytes)),
                bucket,
            )
        };
        let route = self
            .owners
            .get(&bucket)
            .ok_or_else(|| Failure::bad("unknown_annotation_shard"))?;
        Ok((request, route.address.clone()))
    }
}
