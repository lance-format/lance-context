use anyhow::{bail, Context, Result};
use lance_context_ingest::application::{self, Config, Registry};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 2 || args.len() > 3 {
        bail!("usage: lance-context-ingest <bootstrap|ingress|worker|preparer|inspect|annotation-owner|annotation-ingress> CONFIG [TABLE]");
    }
    let config = Config::read(Path::new(&args[1]))?;
    if args[0] == "bootstrap" && args.len() == 2 {
        application::provision(&config).await?;
        println!("{{\"initialized\":true}}");
        return Ok(());
    }
    // Install signal handlers before creating application resources.
    let signals = Signals::new()?;
    if matches!(args[0].as_str(), "annotation-owner" | "annotation-ingress") && args.len() == 2 {
        let options = config
            .http
            .as_ref()
            .context("annotation role requires HTTP configuration")?;
        let listener = tokio::net::TcpListener::bind(options.listen).await?;
        return if args[0] == "annotation-owner" {
            application::annotations::serve_owner(&config, listener, signals.wait()).await
        } else {
            application::annotations::serve_ingress(&config, listener, signals.wait()).await
        };
    }
    let registry = Registry::connect(&config).await?;
    match args[0].as_str() {
        "ingress" if args.len() == 2 => {
            let options = config
                .http
                .as_ref()
                .context("ingress requires HTTP configuration")?;
            let listener = tokio::net::TcpListener::bind(options.listen).await?;
            application::serve_http(&registry, options, listener, signals.wait()).await
        }
        "worker" if args.len() == 2 => {
            application::serve_worker(&registry, &config, signals.wait()).await
        }
        "preparer" if args.len() == 2 => {
            let address = config
                .preparation
                .as_ref()
                .and_then(|p| p.listen)
                .context("preparer requires a listen address")?;
            let listener = tokio::net::TcpListener::bind(address).await?;
            application::serve_preparer(&registry, &config, listener, signals.wait()).await
        }
        "inspect" if args.len() == 3 => {
            let dataset = registry.read(&args[2], None).await?;
            println!(
                "{}",
                serde_json::json!({"table": args[2], "version": dataset.version().version,
                "rows": dataset.count_rows(None).await?})
            );
            Ok(())
        }
        _ => bail!("unknown role or invalid arguments"),
    }
}
#[cfg(unix)]
struct Signals {
    terminate: tokio::signal::unix::Signal,
    interrupt: tokio::signal::unix::Signal,
}
#[cfg(unix)]
impl Signals {
    fn new() -> Result<Self> {
        use tokio::signal::unix::{signal, SignalKind};
        Ok(Self {
            terminate: signal(SignalKind::terminate())?,
            interrupt: signal(SignalKind::interrupt())?,
        })
    }
    async fn wait(mut self) {
        tokio::select! { _ = self.terminate.recv() => {}, _ = self.interrupt.recv() => {} }
    }
}
#[cfg(not(unix))]
struct Signals;
#[cfg(not(unix))]
impl Signals {
    fn new() -> Result<Self> {
        Ok(Self)
    }
    async fn wait(self) {
        let _ = tokio::signal::ctrl_c().await;
    }
}
