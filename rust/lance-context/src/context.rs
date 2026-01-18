use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Context {
    uri: String,
    branch: String,
    next_id: u64,
    entries: Vec<ContextEntry>,
    snapshots: HashMap<String, Snapshot>,
}

impl Context {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            branch: "main".to_string(),
            next_id: 1,
            entries: Vec::new(),
            snapshots: HashMap::new(),
        }
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub fn branch(&self) -> &str {
        &self.branch
    }

    pub fn entries(&self) -> u64 {
        self.entries.len() as u64
    }

    pub fn add(&mut self, role: &str, content: &str, data_type: Option<&str>) -> u64 {
        let entry_id = self.next_id;
        self.next_id += 1;
        self.entries.push(ContextEntry {
            id: entry_id,
            role: role.to_string(),
            data_type: data_type.map(|value| value.to_string()),
            content: content.to_string(),
        });
        entry_id
    }

    pub fn snapshot(&mut self, label: Option<&str>) -> String {
        let id = match label {
            Some(label) if !label.is_empty() => label.to_string(),
            _ => format!("snapshot-{}", self.entries.len()),
        };
        let snapshot = Snapshot {
            id: id.clone(),
            label: label.map(|value| value.to_string()),
            entry_count: self.entries.len() as u64,
            branch: self.branch.clone(),
        };
        self.snapshots.insert(id.clone(), snapshot);
        id
    }

    pub fn fork(&self, branch_name: impl Into<String>) -> Self {
        Self {
            uri: self.uri.clone(),
            branch: branch_name.into(),
            next_id: self.next_id,
            entries: self.entries.clone(),
            snapshots: self.snapshots.clone(),
        }
    }

    pub fn checkout(&mut self, snapshot_id: &str) {
        if let Some(snapshot) = self.snapshots.get(snapshot_id) {
            self.entries.truncate(snapshot.entry_count as usize);
            self.next_id = self.entries.len() as u64 + 1;
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub id: u64,
    pub role: String,
    pub data_type: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub id: String,
    pub label: Option<String>,
    pub entry_count: u64,
    pub branch: String,
}
