//! Cold table discovery, independent of schemas, pending counts and hot caches.
use super::{discover, Mode};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::{stream, StreamExt};
use lance_io::object_store::{ObjectStore, ReadDirOptions};
use object_store::path::Path;

const PAGE_SIZE: usize = 64;
const LIST_TIMEOUT: Duration = Duration::from_secs(30);

/// Revisit immediate `.lance` table directories, including unregistered Context
/// tables and retired tables absent from the hot caches. Opens one object store
/// for the namespace, never a table dataset or a WAL payload. Azure/S3/GCS push
/// the 64-child page limit into storage; local backends may list a whole level
/// internally (the underlying `read_dir_page` fallback).
///
/// Call only from an independently scheduled, cancellable lifecycle consumer.
/// At most one page and four collector admissions are retained. The next page
/// waits for this page's admissions; the collector has its own process bounds.
/// A failed table does not hide the rest of the traversal. An error is returned
/// after visiting the tail, so the owner can report and retry a later round.
pub async fn discover_retired_wal_under_root(uri: &str) -> lance::Result<()> {
    discover_root(Mode::configured()?, uri).await
}

pub(super) async fn discover_root(mode: Mode, uri: &str) -> lance::Result<()> {
    if mode == Mode::Retain {
        return Ok(());
    }
    let (store, root) = tokio::time::timeout(LIST_TIMEOUT, ObjectStore::from_uri(uri))
        .await
        .map_err(|_| lance::Error::io("Retirement namespace open timed out"))??;
    walk_pages(
        |token| {
            let (store, root) = (store.clone(), root.clone());
            async move {
                let page = tokio::time::timeout(
                    LIST_TIMEOUT,
                    store.read_dir_page(
                        root,
                        ReadDirOptions {
                            page_token: token,
                            limit: Some(PAGE_SIZE),
                        },
                    ),
                )
                .await
                .map_err(|_| lance::Error::io("Retirement namespace listing timed out"))??;
                if page.result.common_prefixes.len() + page.result.objects.len() > PAGE_SIZE {
                    return Err(lance::Error::io("Retirement namespace page exceeded limit"));
                }
                // Only immediate directories, never similarly named loose files.
                let tables = page
                    .result
                    .common_prefixes
                    .into_iter()
                    .filter(|p| p.filename().is_some_and(|n| n.ends_with(".lance")))
                    .collect();
                Ok((tables, page.page_token))
            }
        },
        |table| discover(mode, store.clone(), table),
    )
    .await
}

async fn walk_pages<P, PF, V, VF>(page: P, visit: V) -> lance::Result<()>
where
    P: Fn(Option<String>) -> PF,
    PF: Future<Output = lance::Result<(Vec<Path>, Option<String>)>>,
    V: Fn(Path) -> VF,
    VF: Future<Output = lance::Result<()>>,
{
    let mut token = None;
    let failures = AtomicUsize::new(0);
    loop {
        let (tables, next) = page(token.clone()).await?;
        if tables.len() > PAGE_SIZE || (next.is_some() && next == token) {
            return Err(lance::Error::io(
                "Retirement namespace pagination did not advance within bounds",
            ));
        }
        stream::iter(tables)
            .for_each_concurrent(4, |table| {
                let (visit, failures) = (&visit, &failures);
                async move {
                    if let Err(error) = visit(table.clone()).await {
                        failures.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(%table, %error, "Cold WAL retirement discovery deferred");
                    }
                }
            })
            .await;
        token = next;
        // Empty/short pages can still have a continuation token.
        if token.is_none() {
            break;
        }
    }
    match failures.load(Ordering::Relaxed) {
        0 => Ok(()),
        count => Err(lance::Error::io(format!(
            "Retirement discovery deferred for {count} tables"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::sync::Semaphore;

    #[tokio::test]
    async fn retain_root_does_not_open_storage() {
        discover_root(Mode::Retain, "unsupported://no-storage")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn empty_pages_and_poisoned_tables_do_not_hide_cold_tail() {
        let seen = Mutex::new(Vec::new());
        let result = walk_pages(
            |token| async move {
                Ok(match token.as_deref() {
                    None => (vec![], Some("empty".into())),
                    Some("empty") => (vec![Path::from("bad.lance")], Some("tail".into())),
                    Some("tail") => (vec![Path::from("cold.lance")], None),
                    _ => panic!("unexpected cursor"),
                })
            },
            |table| {
                seen.lock().unwrap().push(table.clone());
                async move {
                    if table.as_ref() == "bad.lance" {
                        Err(lance::Error::io("fixture"))
                    } else {
                        Ok(())
                    }
                }
            },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Path::from("bad.lance"), Path::from("cold.lance")]
        );
    }

    #[tokio::test]
    async fn stalled_cursor_is_an_error_not_an_infinite_listing() {
        let result = walk_pages(
            |_| async { Ok((vec![], Some("same".into()))) },
            |_| async { Ok(()) },
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("did not advance"));
    }

    #[tokio::test]
    async fn four_admissions_bound_paging_and_cancellation_drops_waiters() {
        let entered = Arc::new(Semaphore::new(0));
        let lists = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(AtomicUsize::new(0));
        let handle = {
            let (entered, lists, started) = (entered.clone(), lists.clone(), started.clone());
            tokio::spawn(async move {
                walk_pages(
                    |_| {
                        lists.fetch_add(1, Ordering::SeqCst);
                        async {
                            Ok((
                                (0..64).map(|i| Path::from(format!("{i}.lance"))).collect(),
                                Some("next".into()),
                            ))
                        }
                    },
                    |_| {
                        started.fetch_add(1, Ordering::SeqCst);
                        entered.add_permits(1);
                        std::future::pending::<lance::Result<()>>()
                    },
                )
                .await
            })
        };
        tokio::time::timeout(Duration::from_secs(5), entered.acquire_many(4))
            .await
            .unwrap()
            .unwrap()
            .forget();
        assert_eq!(lists.load(Ordering::SeqCst), 1);
        assert_eq!(started.load(Ordering::SeqCst), 4);
        handle.abort();
        assert!(handle.await.unwrap_err().is_cancelled());
        assert_eq!(lists.load(Ordering::SeqCst), 1);
        assert_eq!(started.load(Ordering::SeqCst), 4);
    }
}
