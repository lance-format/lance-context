use super::*;

#[derive(Clone)]
pub struct Owner {
    service: Service,
    authority: p::Authority,
    recovery_cut: u64,
    gate: Gateway,
    reply_limit: usize,
}
impl Owner {
    /// Bind HTTP to this runtime's actual service and fenced publisher identity.
    /// Successful annotation writes in this epoch must await this runtime's
    /// applier, through these routes or its shared Service handle.
    pub fn new(runtime: &Runtime, keys: ReceiptKeys, options: Options) -> Result<Self> {
        let (service, authority, recovery_cut) = runtime.http_binding();
        let ingress = service.ingress().clone();
        let reply_limit = reply_limit(&ingress)?;
        let gate = Gateway::new(
            BTreeMap::from([(ingress.scope.contract.qualified_table.clone(), ingress)]),
            keys,
            options,
        )?;
        Ok(Self {
            service,
            authority,
            recovery_cut,
            gate,
            reply_limit,
        })
    }
    pub fn router(&self) -> Router {
        Router::new()
            .route("/api/v1/health", get(health))
            .route("/api/v1/annotations/{table}/producer", post(open))
            .route("/api/v1/annotations/{table}/update", post(update))
            .route("/api/v1/annotations/{table}/read", post(read))
            .route("/api/v1/annotations/{table}/close", post(close))
            .with_state(self.clone())
    }
    pub fn observation(&self) -> Observation {
        self.gate.observation()
    }
    pub async fn shutdown(&self) {
        self.gate.shutdown().await;
    }

    async fn dispatch(self, table: String, request: Request, operation: Operation) -> Response {
        let version = request.version();
        let guard = match self.gate.enter(matches!(operation, Operation::Update)) {
            Ok(guard) => guard,
            Err(error) => return early(error, request.version()),
        };
        match tokio::spawn(async move {
            let response = self.process(&table, request, operation).await?;
            Ok::<_, Failure>(hold_response(response, guard, None))
        })
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => early(error, version),
            Err(_) => Failure::unavailable("annotation_handler_failed", "unknown").into_response(),
        }
    }

    async fn process(
        &self,
        table: &str,
        request: Request,
        operation: Operation,
    ) -> HttpResult<Response> {
        let ingress = self.service.ingress();
        if table != ingress.scope.contract.qualified_table {
            return Err(Failure {
                status: StatusCode::NOT_FOUND,
                code: "unknown_table",
                outcome: "not_submitted",
            });
        }
        self.service
            .check_owner(&self.authority)
            .await
            .map_err(|_| Failure::unavailable("annotation_owner_unavailable", "not_submitted"))?;
        let options = &self.gate.0.options;
        let response = if matches!(operation, Operation::Update) {
            let (producer, sequence) = producer(ingress, &self.gate.0.keys, &request)?;
            let length = body::length(
                &request,
                ingress.scope.contract.max_record_bytes as usize,
                ARROW_FILE,
            )?;
            let upload = tokio::time::timeout(
                options.admission_timeout,
                self.service.reserve_upload(length),
            )
            .await
            .map_err(|_| Failure::unavailable("memory_capacity", "not_submitted"))?
            .map_err(|_| Failure::bad("invalid_upload_size"))?;
            let bytes = body::read_exact(request, length, options).await?;
            let prepared = self
                .service
                .prepare(upload, bytes, &producer, sequence)
                .await
                .map_err(|_| Failure::bad("invalid_annotation_ipc"))?;
            let completion = self
                .service
                .submit(&prepared, options.body_total_timeout)
                .await
                .map_err(|error| match error.downcast_ref::<AppendFailure>() {
                    Some(AppendFailure::DiskBudgetNotSubmitted) => {
                        Failure::unavailable("wal_disk_capacity", "not_submitted")
                    }
                    // This is a committed WAL decision, not a local capacity
                    // guess or an ambiguous transport failure. Exact retained
                    // requests resolve before the WAL checks capacity.
                    Some(AppendFailure::Capacity) => {
                        Failure::unavailable("wal_capacity", "rejected")
                    }
                    Some(AppendFailure::Sequence { .. } | AppendFailure::ProducerFenced) => {
                        Failure {
                            status: StatusCode::CONFLICT,
                            code: "producer_sequence_rejected",
                            outcome: "rejected",
                        }
                    }
                    _ => Failure::unavailable("annotation_outcome_unknown", "unknown"),
                })?;
            let accepted = AcceptedAppend {
                scope: ingress.scope.clone(),
                position: Receipt {
                    cluster: ingress.scope.cluster,
                    layout: ingress.scope.layout,
                    group: ingress
                        .group(0)
                        .map_err(|_| Failure::bad("invalid_shard"))?,
                    index: completion.index,
                },
                meta: completion.meta.clone(),
                rows: prepared.rows(),
            };
            let receipt = self
                .gate
                .0
                .keys
                .encode(&accepted)
                .map_err(|_| Failure::unavailable("receipt_unavailable", "unknown"))?;
            let status = if matches!(completion.result, MutationResult::Applied) {
                StatusCode::OK
            } else {
                StatusCode::UNPROCESSABLE_ENTITY
            };
            (
                status,
                Json(MutationReply {
                    receipt,
                    completion,
                }),
            )
                .into_response()
        } else {
            let maximum = if matches!(operation, Operation::Read) {
                self.reply_limit
            } else {
                32 << 10
            };
            let bytes = body::read(request, maximum, "application/json", options).await?;
            match operation {
                Operation::Open => {
                    let open: ProducerOpen =
                        serde_json::from_slice(&bytes).map_err(|_| Failure::bad("invalid_open"))?;
                    let producer = self
                        .service
                        .open_producer(open)
                        .await
                        .map_err(|_| Failure::unavailable("open_outcome_unknown", "unknown"))?;
                    Json(Opened {
                        producer: self.gate.0.keys.encode_producer(&producer).map_err(|_| {
                            Failure::unavailable("credential_unavailable", "unknown")
                        })?,
                        generation: producer.generation(),
                        scope: receipt::scope_hash(&ingress.scope)
                            .map_err(|_| Failure::bad("invalid_scope"))?,
                    })
                    .into_response()
                }
                Operation::Read => {
                    let point: PointRequest = serde_json::from_slice(&bytes)
                        .map_err(|_| Failure::bad("invalid_point_read"))?;
                    let value = self
                        .service
                        .get_owned(&point.turn, self.recovery_cut, options.body_total_timeout)
                        .await
                        .map_err(|_| {
                            Failure::unavailable("annotation_read_unavailable", "not_submitted")
                        })?;
                    Json(value).into_response()
                }
                Operation::Close => {
                    let close: CloseRequest = serde_json::from_slice(&bytes)
                        .map_err(|_| Failure::bad("invalid_close"))?;
                    let producer = ingress
                        .restore_producer(&self.gate.0.keys, &close.producer)
                        .map_err(|_| Failure::bad("invalid_producer"))?;
                    let accepted = close
                        .receipt
                        .as_ref()
                        .map(|r| ingress.restore_receipt(&self.gate.0.keys, r))
                        .transpose()
                        .map_err(|_| Failure::bad("invalid_receipt"))?;
                    ingress
                        .close_producer(&producer, accepted.as_ref())
                        .await
                        .map_err(|_| Failure::unavailable("close_outcome_unknown", "unknown"))?;
                    StatusCode::NO_CONTENT.into_response()
                }
                Operation::Update => unreachable!(),
            }
        };
        self.service
            .check_owner(&self.authority)
            .await
            .map_err(|_| {
                Failure::unavailable(
                    "annotation_owner_changed",
                    if operation.is_write() {
                        "unknown"
                    } else {
                        "not_submitted"
                    },
                )
            })?;
        Ok(response)
    }
}

async fn health(State(owner): State<Owner>) -> StatusCode {
    let Ok(_guard) = owner.gate.enter(false) else {
        return StatusCode::SERVICE_UNAVAILABLE;
    };
    if owner.service.check_owner(&owner.authority).await.is_ok() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn open(State(owner): State<Owner>, Path(table): Path<String>, request: Request) -> Response {
    owner.dispatch(table, request, Operation::Open).await
}
async fn update(
    State(owner): State<Owner>,
    Path(table): Path<String>,
    request: Request,
) -> Response {
    owner.dispatch(table, request, Operation::Update).await
}
async fn read(State(owner): State<Owner>, Path(table): Path<String>, request: Request) -> Response {
    owner.dispatch(table, request, Operation::Read).await
}
async fn close(
    State(owner): State<Owner>,
    Path(table): Path<String>,
    request: Request,
) -> Response {
    owner.dispatch(table, request, Operation::Close).await
}
