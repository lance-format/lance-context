use super::*;
use crate::ingest_contract::WritePolicy;
use publication::{Acquire, Authority, Destination, Plan, PlanRequest, SourceRange};
use query::{Outcome as ReadOutcome, Query, Value};
mod closed;

struct Cluster {
    nodes: Vec<Node>,
    root: tempfile::TempDir,
    layout: Arc<Layout>,
    limits: Limits,
    signer: Arc<CatalogSigner>,
    source_signers: BTreeMap<u64, Arc<SourceSigner>>,
    table: Uuid,
    transaction: Uuid,
}
impl Cluster {
    fn new() -> Result<Self> {
        Self::with_plan_limit(128)
    }
    fn with_plan_limit(max_active_plans: u64) -> Result<Self> {
        Self::with_policy(max_active_plans, 100_000)
    }
    fn with_policy(max_active_plans: u64, max_records: u64) -> Result<Self> {
        Self::with_source_policy(max_active_plans, max_records, false)
    }
    fn with_source_policy(max_active_plans: u64, max_records: u64, proofs: bool) -> Result<Self> {
        let root = tempfile::tempdir()?;
        let layout = Arc::new(Layout {
            cluster: Uuid::from_u128(91),
            nodes: (1..=6)
                .map(|i| (i, format!("wal-{i}.example.invalid:7800")))
                .collect(),
            groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![4, 5, 6])]),
            buckets: vec![10, 20],
        });
        let key = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
            .map_err(|_| anyhow::anyhow!("test key generation"))?;
        let signer = Arc::new(CatalogSigner::from_pkcs8(key.as_ref())?);
        let mut source_signers = BTreeMap::new();
        if proofs {
            for group in layout.groups.keys() {
                let key = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                    .map_err(|_| anyhow::anyhow!("test source key generation"))?;
                source_signers.insert(*group, Arc::new(SourceSigner::from_pkcs8(key.as_ref())?));
            }
        }
        let limits = Limits {
            catalog_key: Some(signer.public_key()),
            source_keys: source_signers
                .iter()
                .map(|(g, s)| (*g, s.public_key()))
                .collect(),
            max_active_plans,
            max_records,
            ..Default::default()
        };
        let mut c = Self {
            nodes: vec![],
            root,
            layout,
            limits,
            signer,
            source_signers,
            table: Uuid::from_u128(92),
            transaction: Uuid::from_u128(93),
        };
        c.reopen()?;
        Ok(c)
    }
    fn reopen(&mut self) -> Result<()> {
        self.nodes.clear();
        for id in 1..=6 {
            let mut node = Node::open(
                &self.root.path().join(id.to_string()),
                self.layout.clone(),
                id,
                self.limits.clone(),
            )?;
            if id <= 3 {
                node.set_catalog_signer(self.signer.clone())?;
            }
            for (group, signer) in &self.source_signers {
                if self.layout.groups[group].contains(&id) {
                    node.set_source_signer(*group, signer.clone())?;
                }
            }
            self.nodes.push(node);
        }
        self.nodes[0].campaign(10)?;
        self.nodes[3].campaign(20)?;
        self.pump()?;
        Ok(())
    }
    fn pump(&mut self) -> Result<Events> {
        let mut all = Events::default();
        for _ in 0..200 {
            let mut packets = vec![];
            for node in &mut self.nodes {
                let events = node.poll()?;
                all.completions.extend(events.completions);
                all.reads.extend(events.reads);
                packets.extend(events.messages);
            }
            if packets.is_empty() {
                return Ok(all);
            }
            for packet in packets {
                self.nodes[packet.message.to as usize - 1].step(packet)?;
            }
        }
        anyhow::bail!("pin test pump did not settle")
    }
    fn append(&mut self, table: Uuid, sequence: u64) -> Result<()> {
        self.nodes[3].propose(
            20,
            sequence,
            Batch::new(
                table,
                1,
                Uuid::from_u128(94),
                sequence,
                Bytes::from(vec![sequence as u8; 4096]),
            ),
        )?;
        assert!(self
            .pump()?
            .completions
            .iter()
            .any(|c| matches!(c.outcome, Outcome::Committed(Applied::Accepted { .. }))));
        Ok(())
    }
    fn acquire(&mut self, expected_epoch: u64) -> Result<Authority> {
        self.nodes[0].acquire_publisher(
            100,
            Acquire {
                table: self.table,
                destination: Destination {
                    namespace: Uuid::from_u128(95),
                    dataset: "tables/synthetic".into(),
                },
                ingest: IngestContract {
                    tenant: "test".into(),
                    qualified_table: "synthetic".into(),
                    schema: [7; 32],
                    policy: WritePolicy::ImmutableUniqueIds,
                    max_record_bytes: 1024 * 1024,
                    max_rows: 1024,
                },
                owner: Uuid::from_u128(100 + u128::from(expected_epoch)),
                request: Uuid::from_u128(200 + u128::from(expected_epoch)),
                expected_epoch,
            },
        )?;
        let events = self.pump()?;
        events
            .completions
            .into_iter()
            .find_map(|c| match c.outcome {
                Outcome::Committed(Applied::Publisher(publication::Reply::Acquired(a))) => Some(a),
                _ => None,
            })
            .context("missing publisher completion")
    }
    fn plan(&mut self, mutate: impl FnOnce(&mut SourceRange)) -> Result<u64> {
        self.plan_with_background(mutate, 12)
    }
    fn plan_with_background(
        &mut self,
        mutate: impl FnOnce(&mut SourceRange),
        background: u64,
    ) -> Result<u64> {
        for seq in 1..=background {
            self.append(Uuid::from_u128(999), seq)?;
        }
        self.append(self.table, 1)?;
        self.append(self.table, 2)?;
        self.nodes[3].query(
            20,
            101,
            Query::Accepted {
                after: 0,
                limit: 128,
            },
        )?;
        let reads = self.pump()?.reads;
        let ReadOutcome::Observed {
            value: Value::Accepted(page),
            ..
        } = &reads[0].outcome
        else {
            anyhow::bail!("source query failed");
        };
        let mut range = SourceRange {
            after: 0,
            through: page.through,
            records: page
                .entries
                .iter()
                .filter(|(_, m)| m.stream == self.table)
                .cloned()
                .collect(),
        };
        mutate(&mut range);
        let authority = self.acquire(0)?;
        self.nodes[0].plan_publication(
            102,
            PlanRequest {
                authority,
                plan: Plan {
                    transaction: self.transaction,
                    read_version: 8,
                    sources: BTreeMap::from([(20, range)]),
                    next_group: 10,
                },
            },
        )?;
        self.pump()?
            .completions
            .into_iter()
            .find_map(|c| match c.outcome {
                Outcome::Committed(Applied::Publisher(publication::Reply::Planned { index })) => {
                    Some(index)
                }
                _ => None,
            })
            .context("missing plan completion")
    }
    fn permit(&mut self, index: u64) -> Result<Permit> {
        self.nodes[0].query(
            10,
            103,
            Query::Pin {
                table: self.table,
                transaction: self.transaction,
                plan_index: index,
                source_group: 20,
            },
        )?;
        let events = self.pump()?;
        let permit = events
            .reads
            .into_iter()
            .find_map(|read| match read.outcome {
                ReadOutcome::Observed {
                    value: Value::Pin(Some(permit)),
                    ..
                } => Some(*permit),
                _ => None,
            })
            .context("missing catalog attestation")?;
        // Exercise the serialized boundary: the receiving source has no private key.
        Ok(bincode::deserialize(&bincode::serialize(&permit)?)?)
    }
    fn pin(&mut self, permit: Permit) -> Result<Reply> {
        self.nodes[3].pin_source(104, permit)?;
        self.pump()?
            .completions
            .into_iter()
            .find_map(|c| match c.outcome {
                Outcome::Committed(Applied::Source(reply)) => Some(reply),
                _ => None,
            })
            .context("missing source completion")
    }
}

#[test]
fn disjoint_groups_pin_their_full_prefix_and_recover_original_index() -> Result<()> {
    let mut c = Cluster::new()?;
    let plan = c.plan(|_| {})?;
    let permit = c.permit(plan)?;
    assert!(
        permit.selection.range.through > permit.cut.applied,
        "test did not separate group indices"
    );
    assert!(c.nodes[3].signer.is_none());
    assert!(c.nodes[3].set_catalog_signer(c.signer.clone()).is_err());
    let result = c.pin(permit.clone())?;
    let Reply::Pinned { index } = result else {
        anyhow::bail!("pin rejected");
    };
    assert!(index > permit.selection.range.through);
    c.acquire(1)?;
    let refreshed = c.permit(plan)?;
    assert_ne!(permit.cut, refreshed.cut);
    assert_eq!(c.pin(refreshed)?, result);
    c.append(c.table, 3)?; // Suffix stays outside the immutable pin.
    for n in 0..6 {
        let group = if n < 3 { 10 } else { 20 };
        let image = c.nodes[n]
            .prepare_checkpoint(group, Uuid::from_u128(300 + n as u128))?
            .run()?;
        c.nodes[n].install_checkpoint(image)?;
    }
    c.reopen()?;
    let refreshed = c.permit(plan)?;
    assert_eq!(c.pin(refreshed)?, result);
    for node in &c.nodes[3..] {
        let state = &node.groups[&20].state;
        let pin = &state.pins[&c.table];
        assert_eq!(pin.index, index);
        for (i, meta) in &pin.permit.selection.range.records {
            let row = node.read_accepted(20, *i)?.unwrap();
            assert_eq!(row.meta, *meta);
            assert_eq!(row.body, Bytes::from(vec![meta.sequence as u8; 4096]));
        }
        let captured = state.clone();
        assert!(Arc::ptr_eq(&captured.pins[&c.table], pin));
    }
    Ok(())
}

#[test]
fn maximum_publication_window_survives_catalog_and_source_checkpoints() -> Result<()> {
    let mut c = Cluster::new()?;
    for sequence in 1..=publication::MAX_INPUTS as u64 {
        c.append(c.table, sequence)?;
    }
    let source = &c.nodes[3].groups[&20].state;
    let range = SourceRange {
        after: 0,
        through: source.applied,
        records: source
            .accepted
            .iter()
            .map(|(i, m)| (*i, m.clone()))
            .collect(),
    };
    assert_eq!(range.records.len(), publication::MAX_INPUTS);
    let authority = c.acquire(0)?;
    c.nodes[0].plan_publication(
        102,
        PlanRequest {
            authority,
            plan: Plan {
                transaction: c.transaction,
                read_version: 8,
                sources: BTreeMap::from([(20, range.clone())]),
                next_group: 10,
            },
        },
    )?;
    let index = c
        .pump()?
        .completions
        .into_iter()
        .find_map(|completion| match completion.outcome {
            Outcome::Committed(Applied::Publisher(publication::Reply::Planned { index })) => {
                Some(index)
            }
            _ => None,
        })
        .context("missing maximum-window plan")?;
    let permit = c.permit(index)?;
    assert_eq!(permit.selection.range, range);
    assert!(permit.clone().batch(&c.layout, &c.limits)?.body.len() > 136 * 1024);
    let result = c.pin(permit)?;
    assert!(matches!(result, Reply::Pinned { .. }));
    c.append(c.table, publication::MAX_INPUTS as u64 + 1)?;
    for n in 0..6 {
        let group = if n < 3 { 10 } else { 20 };
        let image = c.nodes[n]
            .prepare_checkpoint(group, Uuid::from_u128(500 + n as u128))?
            .run()?;
        c.nodes[n].install_checkpoint(image)?;
    }
    c.reopen()?;
    let recovered = c.permit(index)?;
    assert_eq!(recovered.selection.range, range);
    assert_eq!(c.pin(recovered)?, result);
    for node in &c.nodes[3..] {
        let state = &node.groups[&20].state;
        assert_eq!(state.pins[&c.table].permit.selection.range, range);
        assert_eq!(state.accepted.len(), publication::MAX_INPUTS + 1);
        for (index, meta) in &state.accepted {
            let record = node
                .read_accepted(20, *index)?
                .context("lost source payload")?;
            assert_eq!(record.meta, *meta);
            assert_eq!(record.body, Bytes::from(vec![meta.sequence as u8; 4096]));
        }
    }
    Ok(())
}

#[test]
fn signed_catalog_selection_cannot_omit_or_change_the_source_prefix() -> Result<()> {
    for mutation in 0..4 {
        let mut c = Cluster::new()?;
        let plan = c.plan(|range| match mutation {
            0 => {
                range.records.remove(0);
            }
            1 => {
                range.records[0].1.digest[0] ^= 1;
            }
            2 => {
                range.through += 10_000;
            }
            _ => {
                range.records[0].0 -= 1;
            }
        })?;
        let permit = c.permit(plan)?;
        assert_eq!(c.pin(permit)?, Reply::Rejected(Rejection::Prefix));
        assert!(c.nodes[3].groups[&20].state.pins.is_empty());
    }
    Ok(())
}

#[test]
fn altered_attestations_and_foreign_keys_never_enqueue_a_pin() -> Result<()> {
    let mut c = Cluster::new()?;
    let plan = c.plan(|_| {})?;
    let permit = c.permit(plan)?;
    let before = c.nodes[3].groups[&20].raw.raft.raft_log.last_index();
    let mut altered = permit.clone();
    altered.selection.range.records.remove(0);
    assert!(c.nodes[3].pin_source(104, altered).is_err());
    for mutate in [
        (|p: &mut Permit| p.cut.cluster = Uuid::from_u128(900)) as fn(&mut Permit),
        |p| p.selection.plan_index += 1,
        |p| p.signature.0[0] ^= 1,
        |p| p.selection.ingest.schema[0] ^= 1,
    ] {
        let mut altered = permit.clone();
        mutate(&mut altered);
        assert!(c.nodes[3].pin_source(104, altered).is_err());
    }
    let mut foreign = c.limits.clone();
    foreign.catalog_key = Some([42; 32]);
    assert!(permit.verify(&c.layout, &foreign, 20).is_err());
    assert_eq!(
        c.nodes[3].groups[&20].raw.raft.raft_log.last_index(),
        before
    );
    assert_eq!(c.nodes[3].status(20)?.pending, 0);
    assert!(matches!(c.pin(permit)?, Reply::Pinned { .. }));
    Ok(())
}

#[test]
fn an_isolated_catalog_cannot_issue_a_new_attestation() -> Result<()> {
    let mut c = Cluster::new()?;
    let plan = c.plan(|_| {})?;
    c.nodes[0].query(
        10,
        105,
        Query::Pin {
            table: c.table,
            transaction: c.transaction,
            plan_index: plan,
            source_group: 20,
        },
    )?;
    let mut unknown = false;
    for _ in 0..40 {
        c.nodes[0].tick()?;
        for read in c.nodes[0].poll()?.reads {
            assert!(matches!(read.outcome, ReadOutcome::Unknown));
            unknown = true;
        }
    }
    assert!(unknown);
    Ok(())
}

#[test]
fn catalog_can_also_be_a_source_and_retry_a_pin_at_capacity() -> Result<()> {
    let mut c = Cluster::with_plan_limit(1)?;
    let authority = c.acquire(0)?;
    let row = Batch::new(
        c.table,
        0,
        Uuid::from_u128(401),
        1,
        Bytes::from(vec![17; 8192]),
    );
    c.nodes[0].propose(10, 400, row.clone())?;
    let index = c
        .pump()?
        .completions
        .into_iter()
        .find_map(|completion| match completion.outcome {
            Outcome::Committed(Applied::Accepted { index }) => Some(index),
            _ => None,
        })
        .context("missing catalog source append")?;
    c.nodes[0].plan_publication(
        401,
        PlanRequest {
            authority,
            plan: Plan {
                transaction: c.transaction,
                read_version: 3,
                next_group: 20,
                sources: BTreeMap::from([(
                    10,
                    SourceRange {
                        after: 0,
                        through: index,
                        records: vec![(index, row.meta.clone())],
                    },
                )]),
            },
        },
    )?;
    let plan_index = c
        .pump()?
        .completions
        .into_iter()
        .find_map(|completion| match completion.outcome {
            Outcome::Committed(Applied::Publisher(publication::Reply::Planned { index })) => {
                Some(index)
            }
            _ => None,
        })
        .context("missing catalog-source plan")?;
    let mut original = None;
    for attempt in 0..2 {
        c.nodes[0].query(
            10,
            402,
            Query::Pin {
                table: c.table,
                transaction: c.transaction,
                plan_index,
                source_group: 10,
            },
        )?;
        let permit = c
            .pump()?
            .reads
            .into_iter()
            .find_map(|read| match read.outcome {
                ReadOutcome::Observed {
                    value: Value::Pin(Some(p)),
                    ..
                } => Some(*p),
                _ => None,
            })
            .context("missing same-group attestation")?;
        c.nodes[0].pin_source(403, permit)?;
        let pinned = c
            .pump()?
            .completions
            .into_iter()
            .find_map(|completion| match completion.outcome {
                Outcome::Committed(Applied::Source(Reply::Pinned { index })) => Some(index),
                _ => None,
            })
            .context("missing same-group pin")?;
        if attempt == 0 {
            original = Some(pinned);
        } else {
            assert_eq!(Some(pinned), original);
        }
        assert_eq!(c.nodes[0].groups[&10].state.pins.len(), 1);
    }
    let state = &c.nodes[0].groups[&10].state;
    state.validate_checkpoint(&c.layout, 10, &c.limits, state.applied)?;
    let mut corrupt = state.clone();
    Arc::make_mut(corrupt.pins.get_mut(&c.table).unwrap())
        .permit
        .signature
        .1[0] ^= 1;
    assert!(corrupt
        .validate_checkpoint(&c.layout, 10, &c.limits, corrupt.applied)
        .is_err());
    let mut wrong_cut = state.clone();
    Arc::make_mut(wrong_cut.pins.get_mut(&c.table).unwrap()).index = index;
    assert!(wrong_cut
        .validate_checkpoint(&c.layout, 10, &c.limits, wrong_cut.applied)
        .is_err());
    Ok(())
}
