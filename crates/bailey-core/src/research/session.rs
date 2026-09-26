use anyhow::{Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use super::{ResearchConfig, journal::Journal};

pub(super) struct Session {
    pub root: PathBuf,
    pub journal: Journal,
    started: Instant,
    maximum: Duration,
}

impl Session {
    pub fn create(config: &ResearchConfig, path: &Path) -> Result<Self> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(path).with_context(|| {
            format!(
                "Session deja existante ou inaccessible : {}",
                path.display()
            )
        })?;
        fs::write(path.join("config.json"), serde_json::to_vec_pretty(config)?)?;
        Ok(Self {
            root: path.to_owned(),
            journal: Journal::create(path)?,
            started: Instant::now(),
            maximum: Duration::from_secs(config.max_minutes * 60),
        })
    }

    pub fn stop_reason(&self) -> Option<&'static str> {
        if self.root.join("STOP").exists() {
            Some("Fichier STOP detecte")
        } else if self.started.elapsed() >= self.maximum {
            Some("Budget de temps atteint")
        } else {
            None
        }
    }

    pub fn wait(&self, seconds: u64) -> bool {
        let wait = Duration::from_secs(seconds);
        let start = Instant::now();
        while start.elapsed() < wait {
            if self.stop_reason().is_some() {
                return false;
            }
            thread::sleep((wait - start.elapsed().min(wait)).min(Duration::from_millis(200)));
        }
        self.stop_reason().is_none()
    }
}
