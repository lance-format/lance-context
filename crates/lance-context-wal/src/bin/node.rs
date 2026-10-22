use anyhow::Result;
use clap::Parser;
use lance_context_wal::service::{Config, Service};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    about = "Independent durable WAL node; configuration and signing identities persist across restarts"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long,default_value_t=2,value_parser=clap::value_parser!(u16).range(1..=64))]
    io_threads: u16,
    #[arg(long,default_value_t=4,value_parser=clap::value_parser!(u16).range(1..=64))]
    blocking_threads: u16,
}
async fn signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! { result=tokio::signal::ctrl_c()=>result?, _=terminate.recv()=>{} }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    let config = Config::read(&args.config)?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(usize::from(args.io_threads))
        .max_blocking_threads(usize::from(args.blocking_threads))
        .build()?
        .block_on(async {
            let mut service = Service::start(config).await?;
            tokio::select! {
                result=signal()=>{
                    // Even signal-handler setup failure must join the running
                    // service before dropping this Tokio runtime.
                    let stopped=service.shutdown().await;
                    result.and(stopped)
                },
                result=service.wait()=>result,
            }
        })
}
