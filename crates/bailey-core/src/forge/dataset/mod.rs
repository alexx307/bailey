//! Corpus pré-tokenisés vérifiables, partitions indépendantes et immuables.
//!
//! Le constructeur conserve la provenance originale et le tokenizer. Les SHA256
//! détectent les corruptions accidentelles, sans constituer une signature d'auteur.
//! Construction pilote en RAM ; lecture soit en RAM, soit par pages dans `stream`.
//! Aucune lecture des shards test pour charger train avec l'un de ces lecteurs.

mod build;
mod duplicate;
mod io;
mod load;
mod manifest;
pub mod stream;

pub use build::build;
pub use load::{load_partition, read_manifest};
pub use manifest::{FileRecord, Manifest, Partition, Shard, TokenizerRecord};

const PARTITIONS: [&str; 3] = ["train", "validation", "test"];

#[cfg(test)]
mod tests;
