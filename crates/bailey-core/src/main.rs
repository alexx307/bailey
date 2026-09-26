use clap::Parser;

fn main() -> anyhow::Result<()> {
    bailey_core::app::execute(bailey_core::app::Cli::parse())
}
