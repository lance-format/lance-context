//! Persistable membership proof for a durable ACK, without retaining its body.
//!
//! Supply persistent random keys from trusted service configuration. All key
//! holders can issue receipts; tokens neither authenticate users nor authorize
//! table access. Resolve and authorize the table before restoring a receipt.
//! Install a new key on all readers before switching issuance, and retain old
//! keys while their receipts must remain verifiable. Tokens do not expire:
//! losing or retiring a key must never authorize deletion of acknowledged data.

use super::{AcceptedAppend, Ingress, Scope};
use anyhow::{ensure, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use lance_context_wal::{placement::Receipt, record::BatchMeta};
use ring::{digest, hmac};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

const VERSION: u8 = 1;
const HEADER: usize = 5;
const TAG: usize = 32;
const MAX_CLAIMS: usize = 8192;
const MAX_BINARY: usize = HEADER + MAX_CLAIMS + TAG;
const MAX_TOKEN: usize = MAX_BINARY.div_ceil(3) * 4;
const DOMAIN: &[u8] = b"lance-context/durable-ingress-receipt/v1";

/// Deliberately not serializable or Debug: key material stays in configuration.
pub struct ReceiptKeys {
    active: u32,
    keys: BTreeMap<u32, hmac::Key>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Claims {
    scope: [u8; 32],
    position: Receipt,
    meta: BatchMeta,
    rows: u32,
}

impl ReceiptKeys {
    pub fn new(active: u32, keys: impl IntoIterator<Item = (u32, [u8; 32])>) -> Result<Self> {
        let mut configured = BTreeMap::new();
        for (id, key) in keys {
            ensure!(
                id != 0 && key != [0; 32],
                "invalid receipt key configuration"
            );
            ensure!(configured.len() < 16, "too many receipt keys");
            ensure!(
                configured
                    .insert(id, hmac::Key::new(hmac::HMAC_SHA256, &key))
                    .is_none(),
                "duplicate receipt key identifier"
            );
        }
        ensure!(
            configured.contains_key(&active),
            "active receipt key is missing"
        );
        Ok(Self {
            active,
            keys: configured,
        })
    }

    pub fn encode(&self, accepted: &AcceptedAppend) -> Result<String> {
        let claims = serde_json::to_vec(&Claims {
            scope: scope_hash(&accepted.scope)?,
            position: accepted.position.clone(),
            meta: accepted.meta.clone(),
            rows: accepted.rows,
        })?;
        ensure!(
            claims.len() <= MAX_CLAIMS,
            "receipt claims exceed protocol limit"
        );
        let mut binary = Vec::with_capacity(HEADER + claims.len() + TAG);
        binary.push(VERSION);
        binary.extend_from_slice(&self.active.to_le_bytes());
        binary.extend_from_slice(&claims);
        let mut context = hmac::Context::with_key(&self.keys[&self.active]);
        context.update(DOMAIN);
        context.update(&binary);
        binary.extend_from_slice(context.sign().as_ref());
        Ok(URL_SAFE_NO_PAD.encode(binary))
    }

    fn decode(&self, token: &str, scope: Arc<Scope>) -> Result<AcceptedAppend> {
        ensure!(
            token.len() <= MAX_TOKEN,
            "receipt token exceeds protocol limit"
        );
        let binary = URL_SAFE_NO_PAD
            .decode(token)
            .context("invalid receipt encoding")?;
        ensure!(
            (HEADER + TAG + 1..=MAX_BINARY).contains(&binary.len()),
            "invalid receipt length"
        );
        ensure!(binary[0] == VERSION, "unsupported receipt version");
        let id = u32::from_le_bytes(binary[1..HEADER].try_into()?);
        let key = self.keys.get(&id).context("receipt key is unavailable")?;
        let end = binary.len() - TAG;
        // Verification is constant-time and precedes parsing any claims. The
        // domain prefix is bounded and contains no caller-controlled length.
        let mut authenticated = Vec::with_capacity(DOMAIN.len() + end);
        authenticated.extend_from_slice(DOMAIN);
        authenticated.extend_from_slice(&binary[..end]);
        hmac::verify(key, &authenticated, &binary[end..])
            .map_err(|_| anyhow::anyhow!("invalid receipt authentication"))?;
        let claims: Claims = serde_json::from_slice(&binary[HEADER..end])?;
        ensure!(claims.scope == scope_hash(&scope)?, "foreign receipt scope");
        Ok(AcceptedAppend {
            scope,
            position: claims.position,
            meta: claims.meta,
            rows: claims.rows,
        })
    }
}

fn scope_hash(scope: &Scope) -> Result<[u8; 32]> {
    let mut hash = digest::Context::new(&digest::SHA256);
    hash.update(b"lance-context/durable-ingress-scope/v1");
    hash.update(&serde_json::to_vec(scope)?);
    Ok(hash.finish().as_ref().try_into()?)
}

impl Ingress {
    /// Restore only after the application resolves and authorizes this binding.
    /// Complete scope comparison includes the table incarnation and contract.
    pub fn restore_receipt(&self, keys: &ReceiptKeys, token: &str) -> Result<AcceptedAppend> {
        let accepted = keys.decode(token, self.scope.clone())?;
        self.check_receipt(&accepted)?;
        Ok(accepted)
    }
}
