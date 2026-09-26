use anyhow::{Context, Result};
use serde_json::json;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

pub(super) struct Journal(File);

impl Journal {
    pub fn create(session: &Path) -> Result<Self> {
        Ok(Self(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(session.join("events.jsonl"))?,
        ))
    }

    pub fn write(&mut self, event: &str, cycle: usize, message: &str) -> Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let entry = json!({"unix": timestamp, "event": event, "cycle": cycle, "message": message});
        serde_json::to_writer(&mut self.0, &entry)?;
        self.0.write_all(b"\n")?;
        self.0.flush().context("Ecriture du journal de recherche")
    }
}
