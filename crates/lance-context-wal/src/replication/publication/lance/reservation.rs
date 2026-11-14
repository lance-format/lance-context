//! Resolve an uncertain reservation without uploading another manifest.
use super::*;

impl ManagedManifest {
    pub(super) async fn reserve(
        &self,
        scope: &ManifestCommit,
        object: ManifestObject,
    ) -> ::lance::Result<Arc<ManifestRecord>> {
        let request = scope.request(object);
        let first = self.catalog.reserve_manifest(request.clone()).await;
        match &first {
            Ok(Outcome::Committed(Applied::Publisher(super::super::Reply::ManifestReserved(
                record,
            )))) => return self.reserved(&request, record.clone()),
            Ok(Outcome::Unknown) | Err(_) => {}
            _ => return Err(error(format!("manifest reservation refused: {first:?}"))),
        }

        // A missing result is not a rejected reservation. Read the same fixed
        // target at a validated quorum cut before considering resubmission.
        if let Some(record) = self.read(Some(scope.target)).await? {
            return self.reserved(&request, record);
        }

        // The catalog checks authority, predecessor and prepared descriptors
        // atomically. Both attempts carry exactly the same immutable object;
        // whichever commits first selects it, and the other is idempotent.
        let second = self.catalog.reserve_manifest(request.clone()).await;
        if let Ok(Outcome::Committed(Applied::Publisher(super::super::Reply::ManifestReserved(
            record,
        )))) = &second
        {
            return self.reserved(&request, record.clone());
        }
        // Even an explicit rejection of the second attempt cannot undo the
        // first unknown outcome. Resolve that original reservation once more.
        if let Some(record) = self.read(Some(scope.target)).await? {
            return self.reserved(&request, record);
        }
        Err(error(format!(
            "manifest reservation outcome unknown; retain original publication: first={first:?}; second={second:?}"
        )))
    }

    fn reserved(
        &self,
        request: &ReserveManifest,
        record: Arc<ManifestRecord>,
    ) -> ::lance::Result<Arc<ManifestRecord>> {
        self.validate_record(&record).map_err(error)?;
        if record.version != request.target
            || record.transaction != request.transaction
            || record.plan_index != request.plan_index
            || record.prepared_index != request.prepared_index
            || record.descriptor_digest != request.descriptor_digest
            || record.object != request.object
            || record.index <= record.prepared_index
        {
            return Err(error(
                "manifest reservation returned a different publication",
            ));
        }
        Ok(record)
    }
}
