//! Corpus pré-tokenisés vérifiables, partitions indépendantes et immuables.
//!
//! Le constructeur conserve la provenance originale et le tokenizer. Les SHA256
//! détectent les corruptions accidentelles, sans constituer une signature d'auteur.
//! Prototype : un fichier source est tokenisé en RAM ; le chargeur retourne toute
//! la partition demandée en RAM. Aucune lecture des shards test pour charger train.

mod build;
mod duplicate;
mod io;
mod load;
mod manifest;

pub use build::build;
pub use load::{load_partition, read_manifest};
pub use manifest::{FileRecord, Manifest, Partition, Shard, TokenizerRecord};

const PARTITIONS: [&str; 3] = ["train", "validation", "test"];

#[cfg(test)]
mod tests;
