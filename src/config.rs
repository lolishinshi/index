use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
pub struct Config {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    #[clap(name = "show-feature")]
    ShowFeature(ShowFeatureCommand),
}

#[derive(Debug, Args)]
pub struct ShowFeatureCommand {
    #[clap()]
    pub image: String,
    #[clap(short, long, default_value = "500")]
    pub num_points: u32,
    #[clap(short, long, default_value = "feature.png")]
    pub output: String,
}
