use clap::Args;
use std::path::PathBuf;
#[derive(Args)]
pub struct ResearchArgs {
    #[arg(long, default_value = "configs/research.json")]
    pub config: PathBuf,
    #[arg(long, default_value = "data/library")]
    pub library: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub init_from: Option<PathBuf>,
    #[arg(long, default_value = "data/core-seed")]
    pub rehearsal: PathBuf,
    #[arg(long, default_value_t = 100)]
    pub steps: usize,
}
