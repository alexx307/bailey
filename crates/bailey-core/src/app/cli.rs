use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Bailey Core : fondation de langage en Rust")]
pub struct Cli {
    #[arg(long, global = true, value_enum, default_value = "cpu")]
    pub device: DeviceChoice,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum DeviceChoice {
    Cpu,
    Cuda,
}

#[derive(Subcommand)]
pub enum Command {
    /// Rechercher et conserver des pages Wikipedia avec leurs sources.
    ResearchTopic { topic: String },
    /// Collecter automatiquement ; --init-from active l'entrainement de candidates.
    Research(super::research_args::ResearchArgs),
    /// Decrire l'architecture et les besoins des poids, sans les allouer.
    Info {
        #[arg(long, default_value = "configs/core-100m.json")]
        config: PathBuf,
    },
    /// Preparer de petits textes francais pour verifier la chaine.
    PrepareSeed {
        #[arg(long, default_value = "data/core-seed")]
        out: PathBuf,
    },
    /// Apprendre notre tokenizer sur le fichier train.txt uniquement.
    TokenizerTrain {
        #[arg(long, default_value = "data/core-seed/train.txt")]
        train: PathBuf,
        #[arg(long, default_value = "data/core-tokenizer")]
        out: PathBuf,
        #[arg(long, default_value_t = 32000)]
        vocab_size: usize,
    },
    /// Verifier forward, tous les gradients et AdamW sur des donnees synthetiques.
    CheckModel {
        #[arg(long, default_value = "configs/core-100m.json")]
        config: PathBuf,
        #[arg(long, default_value_t = 32)]
        sequence: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        #[arg(long)]
        report: PathBuf,
    },
    /// Entrainement next-token ; --tiny sert uniquement a verifier le pipeline.
    Train(TrainArgs),
    /// Mesurer la prediction de tokens sur un fichier reserve.
    Evaluate {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        file: PathBuf,
    },
    /// Completer un texte avec une version sauvegardee.
    Generate {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        tokens: usize,
    },
    /// Console experimentale ; le modele doit avoir appris le dialogue.
    Console {
        #[arg(long)]
        run: PathBuf,
    },
}

#[derive(Args)]
pub struct TrainArgs {
    #[arg(long, default_value = "configs/core-100m.json")]
    pub config: PathBuf,
    #[arg(long, default_value = "data/core-seed")]
    pub data: PathBuf,
    #[arg(long, default_value = "data/core-tokenizer/tokenizer.json")]
    pub tokenizer: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 100)]
    pub steps: usize,
    #[arg(long, default_value_t = 64)]
    pub sequence: usize,
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 0.0003)]
    pub learning_rate: f64,
    #[arg(long, default_value_t = 25)]
    pub eval_every: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long)]
    pub init_from: Option<PathBuf>,
    #[arg(long)]
    pub tiny: bool,
}
