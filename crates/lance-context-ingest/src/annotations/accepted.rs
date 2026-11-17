//! Resolve an uncertain append without changing its identity or payload.
use super::*;

impl Service {
    pub(super) async fn append_observed(
        &self,
        prepared: &PreparedAppend,
    ) -> Result<AcceptedAppend> {
        match self.ingress.append(prepared).await {
            Ok(accepted) => Ok(accepted),
            Err(error)
                if matches!(
                    error.downcast_ref::<AppendFailure>(),
                    Some(AppendFailure::Unknown)
                ) =>
            {
                // Absence does not prove failure: the first proposal may still
                // commit. The WAL deduplicates the identical producer sequence
                // and metadata, so one exact resubmission can resolve it without
                // preparing or applying the mutation a second time.
                match self.observe_accepted(prepared).await {
                    Ok(Some(accepted)) => Ok(accepted),
                    Ok(None) => match self.ingress.append(prepared).await {
                        Ok(accepted) => Ok(accepted),
                        Err(resubmission) => {
                            // Even a definite refusal of this attempt does not
                            // resolve the first attempt. Keep Unknown unless a
                            // fresh observation proves its exact acceptance.
                            let error = error.context(format!(
                                "exact append resubmission failed: {resubmission:#}"
                            ));
                            match self.observe_accepted(prepared).await {
                                Ok(Some(accepted)) => Ok(accepted),
                                Ok(None) => Err(error),
                                Err(observation) => Err(error.context(format!(
                                    "append observation failed: {observation:#}"
                                ))),
                            }
                        }
                    },
                    Err(observation) => {
                        Err(error.context(format!("append observation failed: {observation:#}")))
                    }
                }
            }
            Err(error) => Err(error),
        }
    }

    async fn observe_accepted(&self, prepared: &PreparedAppend) -> Result<Option<AcceptedAppend>> {
        let expected = &prepared.batch.meta;
        let group = self.ingress.group(expected.bucket)?;
        let (cut, value) = reader::observe(
            &self.ingress,
            group,
            query::Query::Producer {
                table: expected.stream,
                bucket: expected.bucket,
                producer: expected.producer,
            },
        )
        .await?;
        let query::Value::Producer(latest) = value else {
            bail!("unexpected append observation");
        };
        let Some((meta, index)) = latest else {
            return Ok(None);
        };
        ensure!(
            meta.stream == expected.stream
                && meta.bucket == expected.bucket
                && meta.producer == expected.producer
                && index > 0
                && index <= cut.applied
                && index < u64::MAX,
            "foreign append observation"
        );
        if meta.sequence < expected.sequence {
            return Ok(None);
        }
        ensure!(
            meta == *expected,
            "append identity was superseded or changed"
        );
        Ok(Some(AcceptedAppend {
            scope: self.ingress.scope.clone(),
            position: Receipt {
                cluster: cut.cluster,
                layout: cut.layout,
                group,
                index,
            },
            meta,
            rows: prepared.rows,
        }))
    }
}
