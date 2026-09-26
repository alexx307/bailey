use std::io::Read;

use anyhow::{Context, Result, ensure};

/// Enforce the size limit even when a server omits or misreports Content-Length.
pub(super) fn read_bounded(reader: impl Read, limit: usize) -> Result<Vec<u8>> {
    let detection_limit = limit
        .checked_add(1)
        .context("Limite de réponse trop grande")?;
    let mut bytes = Vec::new();
    reader
        .take(detection_limit as u64)
        .read_to_end(&mut bytes)
        .context("Lecture de la réponse Wikipédia interrompue")?;
    ensure!(
        bytes.len() <= limit,
        "Réponse Wikipédia supérieure à {limit} octets"
    );
    Ok(bytes)
}
