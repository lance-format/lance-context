//! History retirement is monotonic and carries the original owner fence.
use super::*;

#[derive(Debug)]
pub(crate) struct HistoryUnknown;

impl std::fmt::Display for HistoryUnknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("manifest history retirement outcome unknown")
    }
}
impl std::error::Error for HistoryUnknown {}

pub(crate) async fn retire(
    client: &Client,
    request: p::RetireManifestHistory,
) -> Result<p::HistoryBoundary> {
    let response = client
        .request_routed(Request::Commit(Box::new(Command::RetireHistory(request))))
        .await?;
    match response.reply {
        Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::ManifestHistory(
            boundary,
        )))) => Ok(boundary),
        Reply::Commit(Outcome::Unknown) => Err(HistoryUnknown.into()),
        reply => bail!("manifest history retirement rejected: {reply:?}"),
    }
}
