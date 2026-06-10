use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(name = "lance-context-server")]
#[command(about = "REST API server for lance-context")]
pub struct ServerConfig {
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    #[arg(long, default_value = "3000")]
    pub port: u16,

    #[arg(long, default_value = "./lance-data")]
    pub data_dir: String,
}
