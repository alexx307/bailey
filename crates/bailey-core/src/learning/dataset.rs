use anyhow::{Result, ensure};
use std::{fs, path::Path};

pub fn prepare(export: &Path, rehearsal: &Path, frozen: &Path, out: &Path) -> Result<()> {
    ensure!(!out.exists(), "Corpus de candidate deja present");
    if !frozen.exists() {
        fs::create_dir(frozen)?;
        for (source, destination) in [
            (rehearsal.join("train.txt"), "rehearsal.txt"),
            (rehearsal.join("validation.txt"), "foundation.txt"),
            (export.join("validation.txt"), "web.txt"),
        ] {
            fs::copy(source, frozen.join(destination))?;
        }
    }
    let web = fs::read(export.join("train.txt"))?;
    let old = fs::read(frozen.join("rehearsal.txt"))?;
    ensure!(!web.is_empty() && !old.is_empty(), "Corpus vide");
    ensure!(
        web.len() <= 64 * 1024 * 1024 && old.len() <= 64 * 1024 * 1024,
        "Corpus depasse 64 Mio par source"
    );
    let mut mixed = web.clone();
    mixed.push(b'\n');
    // Au moins la moitie des octets rejouent les lecons precedentes.
    for _ in 0..web.len().div_ceil(old.len()).max(1) {
        mixed.extend_from_slice(&old);
        mixed.push(b'\n');
    }
    fs::create_dir(out)?;
    fs::write(out.join("train.txt"), mixed)?;
    fs::copy(frozen.join("web.txt"), out.join("validation.txt"))?;
    Ok(())
}
