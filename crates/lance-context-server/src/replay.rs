//! Explicit materialization after the caller has retired the previous owner.
//! No ownership transfer, claim reset, or uncertain-operation retry happens here.

use std::str::FromStr;

use crate::{config::ServerConfig, error::AppError, state::AppState};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayTarget {
    Rollout(String),
    Generic(String),
    Context(String),
    Datagen(String),
}

impl FromStr for ReplayTarget {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (kind, name) = value.split_once(':').unwrap_or(("rollout", value));
        AppState::validate_name(name).map_err(|e| format!("invalid replay target: {e:?}"))?;
        match kind {
            "rollout" => Ok(Self::Rollout(name.to_owned())),
            "generic" => Ok(Self::Generic(name.to_owned())),
            "context" => Ok(Self::Context(name.to_owned())),
            "datagen" => Ok(Self::Datagen(name.to_owned())),
            _ => Err(format!("unsupported replay store kind: {kind}")),
        }
    }
}

pub fn validate_config(config: &ServerConfig) -> Result<(), &'static str> {
    if config.replay_own_wal.is_some()
        && config
            .instance_id
            .as_deref()
            .is_none_or(|id| id.trim().is_empty())
    {
        return Err("--replay-own-wal requires a nonempty explicit --instance-id or INSTANCE_ID; HOSTNAME fallback is forbidden");
    }
    Ok(())
}

pub async fn run(state: &AppState, target: &ReplayTarget) -> Result<(), AppError> {
    // Hold the target exclusively through replay and close. Propagate close
    // errors too: AppState::shutdown's log-only policy is unsuitable for a
    // one-shot recovery command's success receipt.
    macro_rules! replay {
        ($open:ident, $name:expr) => {{
            let store = state.$open($name).await?;
            let mut store = store.write().await;
            store
                .replay_and_flush_own_wal()
                .await
                .map_err(AppError::from_lance)?;
            store.close().await.map_err(AppError::from_lance)
        }};
    }
    match target {
        ReplayTarget::Rollout(name) => replay!(get_or_open_rollout_store, name),
        ReplayTarget::Generic(name) => replay!(get_or_open_generic_store, name),
        ReplayTarget::Context(name) => replay!(get_or_open_context_store, name),
        ReplayTarget::Datagen(name) => replay!(get_or_open_datagen_store, name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn replay_target_rejects_ambiguous_or_unsafe_names() {
        assert_eq!("table".parse(), Ok(ReplayTarget::Rollout("table".into())));
        for kind in ["rollout", "generic", "context", "datagen"] {
            assert!(format!("{kind}:table").parse::<ReplayTarget>().is_ok());
        }
        for invalid in [
            "",
            "rollout:",
            "unknown:table",
            "generic:../x",
            "rollout:a:b",
            "/absolute",
        ] {
            assert!(invalid.parse::<ReplayTarget>().is_err(), "{invalid}");
        }
    }

    #[test]
    fn replay_requires_explicit_nonempty_identity() {
        let mut config = ServerConfig::try_parse_from([
            "server",
            "--replay-own-wal",
            "table",
            "--instance-id",
            "owner-15",
        ])
        .unwrap();
        assert!(validate_config(&config).is_ok());
        // Direct validation also forbids HOSTNAME fallback regardless of the
        // invoking test environment's INSTANCE_ID.
        config.instance_id = None;
        assert!(validate_config(&config).is_err());
        config.instance_id = Some("  ".into());
        assert!(validate_config(&config).is_err());
        config.replay_own_wal = None;
        assert!(validate_config(&config).is_ok());
    }

    #[tokio::test]
    async fn replay_missing_target_is_an_error_without_creating_it() {
        let dir = tempfile::tempdir().unwrap();
        let config = ServerConfig::try_parse_from([
            "server",
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--instance-id",
            "test-owner",
        ])
        .unwrap();
        let state = AppState::new(config).await.unwrap();
        for kind in ["rollout", "generic", "context", "datagen"] {
            let target = format!("{kind}:absent").parse().unwrap();
            assert!(run(&state, &target).await.is_err(), "{kind}");
        }
        assert!(!dir.path().join("absent.lance").exists());
        assert!(!dir.path().join("absent.rollout.lance").exists());
        assert!(!dir.path().join("absent.generic.lance").exists());
        assert!(!dir.path().join("absent.datagen.lance").exists());
    }
}
