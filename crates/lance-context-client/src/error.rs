#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("API error ({status}): [{code}] {message}")]
    Api {
        status: u16,
        code: String,
        message: String,
    },
}
