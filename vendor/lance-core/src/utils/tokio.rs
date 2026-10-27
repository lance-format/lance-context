// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, atomic};
use std::time::Duration;

use futures::{Future, FutureExt};
use tokio::runtime::{Builder, Runtime};
use tracing::Span;

tokio::task_local! { static CPU_TASK_SCOPE: Arc<CpuTaskScope>; }
thread_local! {
    // Blocking closures do not inherit Tokio task locals. Propagate the same
    // scope explicitly so nested spawn_cpu calls remain owned by their root.
    static BLOCKING_CPU_TASK_SCOPE: RefCell<Option<Arc<CpuTaskScope>>> = const { RefCell::new(None) };
}

#[derive(Debug, Default)]
struct CpuTaskScope {
    active: atomic::AtomicUsize,
    changed: tokio::sync::Notify,
}

impl CpuTaskScope {
    async fn drain(&self) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.active.load(Ordering::Acquire) == 0 {
                return;
            }
            changed.await;
        }
    }
}

/// A pending operation's membership in the current owned task group. Retain
/// this on individual requests when a shared service batches multiple callers;
/// do not attribute an entire shared service to its first caller's group.
#[derive(Debug)]
pub struct TaskScopeToken(Arc<CpuTaskScope>);
impl Drop for TaskScopeToken {
    fn drop(&mut self) {
        if self.0.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.changed.notify_waiters();
        }
    }
}

/// Retain the current task group until this token is dropped. Unscoped callers
/// receive `None` and keep their original lifetime behavior.
pub fn retain_task_scope() -> Option<TaskScopeToken> {
    let scope = CPU_TASK_SCOPE
        .try_with(Arc::clone)
        .ok()
        .or_else(|| BLOCKING_CPU_TASK_SCOPE.with(|slot| slot.borrow().clone()))?;
    scope.active.fetch_add(1, Ordering::AcqRel);
    Some(TaskScopeToken(scope))
}

#[pin_project::pin_project]
struct ScopedTask<F> {
    // Rust drops fields in declaration order. Even cancellation before the
    // first poll must destroy captured payloads before releasing membership.
    #[pin]
    work: F,
    _active: Option<TaskScopeToken>,
}

impl<F: Future> Future for ScopedTask<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.project().work.poll(cx)
    }
}

/// Capture task ownership before submitting an async child to any executor.
/// The returned future propagates the task-local group while it is polled and
/// releases membership only when the actual future finishes or is dropped.
/// This preserves third-party abort-on-drop task wrappers: requesting abort is
/// not completion, but actually dropping their future is.
pub fn scope_task<F: Future>(work: F) -> impl Future<Output = F::Output> {
    let active = retain_task_scope();
    let scope = active.as_ref().map(|guard| guard.0.clone());
    // Construct TaskLocalFuture before submission so its drop also restores
    // context when a queued child is cancelled before its very first poll.
    let work = match scope {
        Some(scope) => futures::future::Either::Left(CPU_TASK_SCOPE.scope(scope, work)),
        None => futures::future::Either::Right(work),
    };
    ScopedTask {
        work,
        _active: active,
    }
}

/// Spawn a Tokio child with the current group's lifetime and nested context.
/// With no current group, output, panic, and cancellation behavior are unchanged.
pub fn spawn_scoped<F>(work: F) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    tokio::spawn(scope_task(work))
}

struct ScopedBlocking<F> {
    // Drop captured payloads before membership if cancelled while still queued.
    work: Option<F>,
    active: Option<TaskScopeToken>,
}

impl<F, R> ScopedBlocking<F>
where
    F: FnOnce() -> R,
{
    fn run(mut self) -> R {
        let previous = BLOCKING_CPU_TASK_SCOPE
            .with(|slot| slot.replace(self.active.as_ref().map(|guard| guard.0.clone())));
        let _blocking_scope = BlockingScopeGuard(previous);
        (self.work.take().unwrap())()
    }
}

/// Submit blocking IO to the ordinary Tokio pool while retaining its scope
/// until the actual closure exits. Dropping its JoinHandle cannot stop a
/// running closure. Pool selection and JoinHandle panic semantics are unchanged.
pub fn spawn_blocking_scoped<F, R>(work: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let task = ScopedBlocking {
        work: Some(work),
        active: retain_task_scope(),
    };
    tokio::task::spawn_blocking(move || task.run())
}

struct BlockingScopeGuard(Option<Arc<CpuTaskScope>>);
impl Drop for BlockingScopeGuard {
    fn drop(&mut self) {
        BLOCKING_CPU_TASK_SCOPE.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

/// Join all CPU closures and scoped async tasks submitted while polling `work`, including children
/// whose handles were discarded on an early error and nested CPU submissions.
/// The original output or panic is returned only after those closures exit.
/// Unscoped `spawn_cpu` calls retain their existing behavior.
///
/// This future must itself be driven to completion: cancelling it cannot stop
/// a blocking closure. Resource owners should run it in a separately owned,
/// drained task and retain their allocation guard outside `work`. Arbitrary
/// `tokio::spawn` tasks do not inherit this scope: use [`spawn_scoped`] or wrap
/// the submitted future with [`scope_task`]. Shared services can use one
/// [`retain_task_scope`] token per request instead of inheriting one caller.
pub async fn join_cpu_tasks<F: Future>(work: F) -> F::Output {
    let scope = Arc::new(CpuTaskScope::default());
    let result = CPU_TASK_SCOPE
        .scope(
            scope.clone(),
            std::panic::AssertUnwindSafe(work).catch_unwind(),
        )
        .await;
    // Root submission has ended. Nested tasks register before their parent's
    // guard exits, so active cannot reach zero while a child can still submit.
    scope.drain().await;
    match result {
        Ok(output) => output,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

/// We cache the call to num_cpus::get() because:
///
/// 1. It shouldn't change during the lifetime of the program
/// 2. It's a relatively expensive call (requires opening several files and examining them)
static NUM_COMPUTE_INTENSIVE_CPUS: LazyLock<usize> =
    LazyLock::new(calculate_num_compute_intensive_cpus);

pub fn get_num_compute_intensive_cpus() -> usize {
    *NUM_COMPUTE_INTENSIVE_CPUS
}

fn calculate_num_compute_intensive_cpus() -> usize {
    if let Ok(raw) = std::env::var("LANCE_CPU_THREADS") {
        return parse_env_usize("LANCE_CPU_THREADS", &raw, 1).unwrap_or_else(|e| panic!("{e}"));
    }

    let cpus = num_cpus::get();

    if cpus <= *IO_CORE_RESERVATION {
        // If the user is not setting a custom value for LANCE_IO_CORE_RESERVATION then we don't emit
        // a warning because they're just on a small machine and there isn't much they can do about it.
        if cpus > 2 {
            log::warn!(
                "Number of CPUs is less than or equal to the number of IO core reservations. \
                This is not a supported configuration. using 1 CPU for compute intensive tasks."
            );
        }
        return 1;
    }

    num_cpus::get() - *IO_CORE_RESERVATION
}

/// Parse an integer environment variable, rejecting values below `min`.
///
/// The error names the variable, so a bad value is diagnosable instead of
/// surfacing as a bare `ParseIntError` or, for `LANCE_CPU_THREADS=0`, a panic
/// deep inside tokio's `max_blocking_threads`.
fn parse_env_usize(name: &str, raw: &str, min: usize) -> Result<usize, String> {
    let value: usize = raw
        .trim()
        .parse()
        .map_err(|e| format!("environment variable {name} must be an integer, got {raw:?}: {e}"))?;
    if value < min {
        return Err(format!(
            "environment variable {name} must be at least {min}, got {value}"
        ));
    }
    Ok(value)
}

/// Number of CPU cores held back for I/O and control tasks.
///
/// Overridable via the `LANCE_IO_CORE_RESERVATION` environment variable;
/// defaults to `2` when unset. `0` is allowed (reserve nothing);
/// [`get_num_compute_intensive_cpus`] subtracts this from the core count to
/// size the compute pool. A non-integer value panics on first access with an
/// error naming the variable.
pub static IO_CORE_RESERVATION: LazyLock<usize> =
    LazyLock::new(|| match std::env::var("LANCE_IO_CORE_RESERVATION") {
        Ok(raw) => {
            parse_env_usize("LANCE_IO_CORE_RESERVATION", &raw, 0).unwrap_or_else(|e| panic!("{e}"))
        }
        Err(_) => 2,
    });

fn create_runtime() -> Runtime {
    Builder::new_multi_thread()
        .thread_name("lance-cpu")
        .max_blocking_threads(get_num_compute_intensive_cpus())
        .worker_threads(1)
        // keep the thread alive "forever"
        .thread_keep_alive(Duration::from_secs(u64::MAX))
        .build()
        .unwrap()
}

static CPU_RUNTIME: atomic::AtomicPtr<Runtime> = atomic::AtomicPtr::new(std::ptr::null_mut());

static RUNTIME_INSTALLED: atomic::AtomicBool = atomic::AtomicBool::new(false);

static ATFORK_INSTALLED: atomic::AtomicBool = atomic::AtomicBool::new(false);

fn global_cpu_runtime() -> &'static Runtime {
    loop {
        let ptr = CPU_RUNTIME.load(Ordering::SeqCst);
        if !ptr.is_null() {
            // SAFETY: `ptr` was produced by `Box::into_raw` below and is only ever
            // reset to null by `atfork_tokio_child` in the forked child (single-
            // threaded, async-signal context). The `Box` is never reclaimed, so the
            // `Runtime` lives for the rest of the process.
            return unsafe { &*ptr };
        }
        if !RUNTIME_INSTALLED.fetch_or(true, Ordering::SeqCst) {
            break;
        }
        std::thread::yield_now();
    }
    if !ATFORK_INSTALLED.fetch_or(true, Ordering::SeqCst) {
        install_atfork();
    }
    let new_ptr = Box::into_raw(Box::new(create_runtime()));
    CPU_RUNTIME.store(new_ptr, Ordering::SeqCst);
    // SAFETY: `new_ptr` was just obtained from `Box::into_raw`, so it is non-null,
    // aligned, and points to a live `Runtime` that is never reclaimed.
    unsafe { &*new_ptr }
}

/// After a fork() operation, force re-creation of the BackgroundExecutor. Note: this function
/// runs in "async-signal context" which means that we can't (safely) do much here.
extern "C" fn atfork_tokio_child() {
    CPU_RUNTIME.store(std::ptr::null_mut(), Ordering::SeqCst);
    RUNTIME_INSTALLED.store(false, Ordering::SeqCst);
}

#[cfg(not(windows))]
fn install_atfork() {
    unsafe { libc::pthread_atfork(None, None, Some(atfork_tokio_child)) };
}

#[cfg(windows)]
fn install_atfork() {}

/// Spawn a CPU intensive task
///
/// This task will be put onto a thread pool dedicated for CPU-intensive work
/// This keeps the tokio thread pool free so that we can always be ready to service
/// cheap I/O & control requests.
///
/// This can also be used to convert a big chunk of synchronous work into a future
/// so that it can be run in parallel with something like StreamExt::buffered()
///
/// # Only hand over substantial CPU work
///
/// Dispatching to the pool has real overhead (a `spawn_blocking` hop plus a oneshot
/// channel round trip). As a rule of thumb the closure should be expected to do at
/// least ~100µs of CPU work; below that the thread-pool overhead is likely to
/// outweigh any parallelism benefit, and the work is better left inline.
///
/// # The task must never wait on anything
///
/// The CPU pool is sized to [`get_num_compute_intensive_cpus`], which is
/// `max(1, num_cpus - LANCE_IO_CORE_RESERVATION)`. On a big host that is plenty of
/// workers (e.g. 62 on a 64-core box), but in resource-constrained environments it can
/// collapse to a **single blocking thread** — on machines with `<= 3` visible CPUs
/// (1-vCPU VMs, CI runners, CPU-limited Kubernetes pods) the pool has exactly one
/// worker. A closure passed to `spawn_cpu` occupies one of these threads for its entire
/// lifetime, including any time it spends *parked*. So the closure must only consume
/// CPU and return; it must
/// **never** block, wait, or park. Concretely, the closure must not, directly or
/// transitively:
///
/// * **No channels** — no blocking send/recv (`send_blocking`, blocking `recv`, etc.).
///   A full/empty channel parks the thread, and whatever would drain/fill the channel
///   may need the same pool to run.
/// * **No I/O** — no file, network, or object-store reads/writes, and no disk spills.
///   I/O parks the thread while making no progress on CPU work.
/// * **No locks** — no acquiring a contended lock (or any lock that is held across an
///   `.await` elsewhere). Waiting for the lock parks the thread.
/// * **No `block_on` / `.blocking_*`** — never drive or wait on another async task
///   from inside the closure.
///
/// If any of these hold, the parked thread can starve the exact work that would
/// unblock it, deadlocking the whole pool with no timeout and no error — a silent
/// hang at 0% CPU. (See <https://github.com/lancedb/lance/pull/7423>.) When work
/// needs to wait on a channel/lock/I/O, keep the waiting in an async task and only
/// hand the pure-CPU portion to `spawn_cpu`, e.g. build each batch with `spawn_cpu`
/// and dispatch it with `tx.send(batch).await` in the surrounding async code.
pub fn spawn_cpu<
    E: std::error::Error + Send + 'static,
    F: FnOnce() -> std::result::Result<R, E> + Send + 'static,
    R: Send + 'static,
>(
    func: F,
) -> impl Future<Output = std::result::Result<R, E>> {
    // Propagate the current span into the task
    let span = Span::current();
    let active = retain_task_scope();
    let handle = global_cpu_runtime().spawn_blocking(move || {
        let _active = active;
        let previous = BLOCKING_CPU_TASK_SCOPE
            .with(|slot| slot.replace(_active.as_ref().map(|guard| guard.0.clone())));
        let _blocking_scope = BlockingScopeGuard(previous);
        let _span_guard = span.enter();
        func()
    });
    // Awaited through the join handle, not a result channel: a panic in `func`
    // arrives as a `JoinError` still carrying its payload, so resuming it
    // re-raises the original panic in the caller. Reporting the closure's
    // outcome over a channel instead loses that -- the sender drops unsent and
    // every panic in any `spawn_cpu` closure surfaces identically as an opaque
    // `RecvError`, pointing here rather than at the fault.
    handle.map(|res| match res {
        Ok(result) => result,
        Err(join_error) => match join_error.try_into_panic() {
            Ok(panic) => std::panic::resume_unwind(panic),
            // The CPU runtime outlives every caller, so its tasks are not
            // cancelled out from under one.
            Err(join_error) => panic!("spawn_cpu task failed: {join_error}"),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panic in the closure must reach the caller intact.
    ///
    /// Reporting the closure's outcome over a channel loses it: the sender
    /// drops unsent and the caller can only see an opaque receive error, so
    /// every panic in every `spawn_cpu` closure looks the same.
    #[tokio::test]
    async fn spawn_cpu_reraises_the_closure_panic() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let joined = tokio::spawn(async {
            spawn_cpu(|| -> std::result::Result<(), std::io::Error> {
                panic!("the original message")
            })
            .await
        })
        .await;
        std::panic::set_hook(hook);

        let payload = joined
            .expect_err("the closure's panic propagates to the caller")
            .into_panic();
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .expect("the original payload survives");
        assert_eq!(message, "the original message");
    }

    // The env vars feed process-global `LazyLock`s that read once and are read
    // in parallel by other tests, so the pure parser is tested directly rather
    // than by mutating the environment.

    #[test]
    fn parses_valid_value_and_trims_surrounding_whitespace() {
        assert_eq!(parse_env_usize("VAR", "8", 1).unwrap(), 8);
        assert_eq!(parse_env_usize("VAR", " 8 ", 1).unwrap(), 8);
    }

    #[test]
    fn rejects_non_integer_naming_the_variable() {
        let err = parse_env_usize("LANCE_CPU_THREADS", "abc", 1).unwrap_err();
        assert!(err.contains("LANCE_CPU_THREADS"), "{err}");
        assert!(err.contains("must be an integer"), "{err}");
    }

    #[test]
    fn rejects_value_below_minimum() {
        // LANCE_CPU_THREADS=0 parses fine but would panic in tokio's
        // max_blocking_threads(0); the minimum stops it at the boundary.
        let err = parse_env_usize("LANCE_CPU_THREADS", "0", 1).unwrap_err();
        assert!(err.contains("at least 1"), "{err}");
    }

    #[test]
    fn allows_zero_when_minimum_is_zero() {
        // LANCE_IO_CORE_RESERVATION=0 is valid: no cores reserved for IO.
        assert_eq!(parse_env_usize("VAR", "0", 0).unwrap(), 0);
    }
}
