//! Bound the WAL prefix represented by an active annotation generation.
use super::*;
use lance_context_wal::replication::publication as p;

pub(super) struct Window {
    pub after: u64,
    pub records: Vec<(u64, BatchMeta)>,
    pub bytes: u64,
    max_records: usize,
    max_bytes: u64,
}

impl Window {
    pub fn new(after: u64, max_bytes: u64) -> Self {
        Self {
            after,
            records: Vec::new(),
            bytes: 0,
            max_records: p::MAX_INPUTS,
            max_bytes,
        }
    }

    pub fn admits(&self, meta: &BatchMeta) -> bool {
        self.records.len() < self.max_records
            && meta.body_bytes <= self.max_bytes.saturating_sub(self.bytes)
    }

    pub fn record(&mut self, index: u64, meta: BatchMeta) {
        self.bytes += meta.body_bytes;
        self.records.push((index, meta));
    }

    pub fn freeze(&mut self, through: u64) -> p::SourceRange {
        let range = p::SourceRange {
            after: self.after,
            through,
            records: std::mem::take(&mut self.records),
        };
        self.after = through;
        self.bytes = 0;
        range
    }
}

impl Driver {
    /// Configure before running the applier. Even repeated updates of one turn
    /// must fit a bounded publication plan. Reaching this limit pauses apply
    /// until the publisher freezes the current generation; it is not rejection.
    pub fn with_publication_limits(self, records: usize, bytes: u64) -> Result<Self> {
        let ingress = &self.service.ingress;
        ensure!(
            (1..=p::MAX_INPUTS).contains(&records)
                && bytes >= ingress.scope.contract.max_record_bytes as u64
                && bytes <= ingress.client.max_plan_source_bytes(),
            "invalid annotation source window limits"
        );
        {
            let mut window = self.service.shared.window.lock().unwrap();
            ensure!(
                window.records.is_empty(),
                "annotation application already started"
            );
            window.max_records = records;
            window.max_bytes = bytes;
        }
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct SourceObservation {
    pub after: u64,
    pub records: usize,
    pub bytes: u64,
}

impl Service {
    pub fn source_observation(&self) -> SourceObservation {
        let window = self.shared.window.lock().unwrap();
        SourceObservation {
            after: window.after,
            records: window.records.len(),
            bytes: window.bytes,
        }
    }
}
