mod cmd;
mod config;
mod features;
mod utils;

use anyhow::Result;
use clap::Parser;
use cmd::cmd_show_feature;
use config::*;

async fn run() -> Result<()> {
    let config = Config::parse();
    match config.command {
        Command::ShowFeature(cmd) => {
            cmd_show_feature(cmd)?;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("Error: {}", err);
    }
}
