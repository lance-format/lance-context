//! Shared helpers for dataset roots that may be local paths or object-store URIs.

use std::io;
use std::path::Path;

/// Maximum accepted length for a context or rollout store name.
pub const MAX_STORE_NAME_LEN: usize = 128;

/// Join one dataset child onto a local directory or object-store URI.
#[must_use]
pub fn join_uri(base: &str, child: &str) -> String {
    if has_uri_scheme(base) {
        format!("{}/{}", base.trim_end_matches('/'), child)
    } else {
        Path::new(base).join(child).to_string_lossy().to_string()
    }
}

/// Create `data_dir` when it is a local filesystem path.
///
/// Object-store roots such as `s3://`, `gs://`, and `az://` must never be
/// passed to `std::fs`, where they would be interpreted as relative paths.
pub fn create_local_dir_if_needed(data_dir: &str) -> io::Result<()> {
    if has_uri_scheme(data_dir) {
        return Ok(());
    }
    std::fs::create_dir_all(data_dir)
}

/// Validate a logical context/rollout store name before using it in a URI.
///
/// Names are intentionally limited to a portable single-segment form so they
/// cannot escape `data_dir` or acquire platform-specific path semantics.
pub fn validate_store_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("store name must not be empty".to_string());
    }

    let mut chars = name.chars();
    let first = chars.next().expect("non-empty checked above");
    if !(first.is_ascii_alphanumeric() || first == '_') {
        return Err("store name must start with an ASCII letter, digit, or '_'".to_string());
    }
    if !chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')) {
        return Err(
            "store name may contain only ASCII letters, digits, '.', '-', and '_'".to_string(),
        );
    }
    if matches!(name, "_registry" | "_stats") {
        return Err(format!("store name '{name}' is reserved"));
    }
    if name.len() > MAX_STORE_NAME_LEN {
        return Err(format!(
            "store name must be at most {MAX_STORE_NAME_LEN} characters"
        ));
    }
    Ok(())
}

fn has_uri_scheme(value: &str) -> bool {
    let Some((scheme, _)) = value.split_once("://") else {
        return false;
    };
    let mut chars = scheme.chars();
    matches!(chars.next(), Some(ch) if ch.is_ascii_alphabetic())
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_local_paths_and_object_store_uris() {
        assert_eq!(
            join_uri("/data/rollouts", "exp.rollout.lance"),
            "/data/rollouts/exp.rollout.lance"
        );
        assert_eq!(
            join_uri("relative/data/", "exp.rollout.lance"),
            "relative/data/exp.rollout.lance"
        );
        assert_eq!(
            join_uri("s3://bucket/prefix/", "exp.rollout.lance"),
            "s3://bucket/prefix/exp.rollout.lance"
        );
        assert_eq!(
            join_uri("gs://bucket", "_registry.rollout.lance"),
            "gs://bucket/_registry.rollout.lance"
        );
    }

    #[test]
    fn recognizes_uri_schemes_without_treating_local_paths_as_uris() {
        assert!(has_uri_scheme("s3://bucket/prefix"));
        assert!(has_uri_scheme("az://container/prefix"));
        assert!(has_uri_scheme("file:///tmp/data"));
        assert!(!has_uri_scheme("/tmp/data"));
        assert!(!has_uri_scheme("./data"));
        assert!(!has_uri_scheme("experiment:data"));
    }

    #[test]
    fn creates_only_local_directories() {
        let root = tempfile::TempDir::new().unwrap();
        let nested = root.path().join("nested/data");
        create_local_dir_if_needed(nested.to_str().unwrap()).unwrap();
        assert!(nested.is_dir());

        create_local_dir_if_needed("s3://bucket/prefix").unwrap();
        create_local_dir_if_needed("gs://bucket/prefix").unwrap();
    }

    #[test]
    fn validates_portable_single_segment_store_names() {
        for name in [
            "exp-alpha",
            "conntest_lianghongxu",
            "run.2026_07-15",
            "_private",
            "7",
        ] {
            validate_store_name(name).unwrap();
        }

        for name in [
            "",
            "../outside",
            "nested/store",
            r"nested\store",
            "_registry",
            "has space",
            "experiment:name",
        ] {
            assert!(validate_store_name(name).is_err(), "{name}");
        }
        assert!(validate_store_name(&"a".repeat(MAX_STORE_NAME_LEN)).is_ok());
        assert!(validate_store_name(&"a".repeat(MAX_STORE_NAME_LEN + 1)).is_err());
    }
}
