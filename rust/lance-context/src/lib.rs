//! Core types for the lance-context storage layer.

#[derive(Debug, Clone)]
pub struct Context {
    uri: String,
    branch: String,
    entries: u64,
}

impl Context {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            branch: "main".to_string(),
            entries: 0,
        }
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub fn branch(&self) -> &str {
        &self.branch
    }

    pub fn entries(&self) -> u64 {
        self.entries
    }

    pub fn add(&mut self, _role: &str, _content: &str, _data_type: Option<&str>) {
        self.entries += 1;
    }

    pub fn snapshot(&self, label: Option<&str>) -> String {
        match label {
            Some(label) if !label.is_empty() => label.to_string(),
            _ => format!("snapshot-{}", self.entries),
        }
    }

    pub fn fork(&self, branch_name: impl Into<String>) -> Self {
        Self {
            uri: self.uri.clone(),
            branch: branch_name.into(),
            entries: self.entries,
        }
    }

    pub fn checkout(&mut self, _snapshot_id: &str) {}
}
