/// Base-table key index policy used by WAL merge and scheduled indexing.
/// All writers of a table should use the same policy to avoid rebuilding the
/// index back and forth during maintenance. Opening a store does not rebuild it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum KeyIndexType {
    /// Exact key lookup; preserves the existing default.
    #[default]
    Btree,
    /// Compact zone summaries, with key-only predicate deletion during merge.
    Zonemap,
}

impl KeyIndexType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Btree => "btree",
            Self::Zonemap => "zonemap",
        }
    }
}
