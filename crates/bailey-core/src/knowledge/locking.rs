use anyhow::{Context, Result};
use fs2::FileExt;
use std::{
    fs::{self, File, OpenOptions},
    path::Path,
};

/// Process-wide filesystem lock; closing the handle releases it even after a crash.
pub struct LibraryLock {
    _file: File,
}

impl LibraryLock {
    pub fn acquire(root: &Path) -> Result<Self> {
        fs::create_dir_all(root)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("library.lock"))?;
        file.try_lock_exclusive()
            .context("Bibliotheque occupee par un autre processus")?;
        Ok(Self { _file: file })
    }
}
