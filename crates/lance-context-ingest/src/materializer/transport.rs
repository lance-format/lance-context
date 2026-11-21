//! A lost RPC exchange says neither committed nor rejected.
//!
//! Call only on WAL client errors, not file or blob IO. This classification
//! does not authorize replay: each control protocol must still reconcile its
//! original identity under fresh authority. Business mutations are unchanged.
pub(super) fn uncertain(error: &anyhow::Error) -> bool {
    use std::io::ErrorKind;

    // Certificate/identity failures, malformed frames, foreign scopes and
    // application rejections are not transient transport uncertainty.
    error.is::<tokio::time::error::Elapsed>()
        || error.downcast_ref::<std::io::Error>().is_some_and(|e| {
            matches!(
                e.kind(),
                ErrorKind::UnexpectedEof
                    | ErrorKind::BrokenPipe
                    | ErrorKind::ConnectionReset
                    | ErrorKind::ConnectionAborted
                    | ErrorKind::NotConnected
                    | ErrorKind::ConnectionRefused
                    | ErrorKind::TimedOut
            )
        })
}
