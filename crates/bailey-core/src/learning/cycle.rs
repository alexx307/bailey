use super::{
    dataset,
    selection::{self, Scores},
};
use crate::{
    research::ResearchConfig,
    training::{self, checkpoint, evaluation},
};
use anyhow::{Result, ensure};
use candle_core::Device;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub struct Learner {
    active: Option<PathBuf>,
    rehearsal: PathBuf,
    session: PathBuf,
    steps: usize,
    initial: Option<Scores>,
    started: Instant,
    budget: Duration,
}
impl Learner {
    pub fn new(
        active: Option<PathBuf>,
        rehearsal: &Path,
        session: &Path,
        steps: usize,
        config: &ResearchConfig,
    ) -> Result<Self> {
        config.validate()?;
        ensure!((1..=5000).contains(&steps), "Etapes attendues : 1..5000");
        if let Some(run) = &active {
            checkpoint::read_config(run)?;
        }
        Ok(Self {
            active,
            rehearsal: rehearsal.to_owned(),
            session: session.to_owned(),
            steps,
            initial: None,
            started: Instant::now(),
            budget: Duration::from_secs(config.max_minutes * 60),
        })
    }
    fn stopped(&self) -> bool {
        self.session.join("STOP").exists() || self.started.elapsed() >= self.budget
    }
    pub fn cycle(&mut self, export: &Path, cycle: usize, device: &Device) -> Result<()> {
        let Some(active) = self.active.clone() else {
            return Ok(());
        };
        if self.stopped() {
            return Ok(());
        }
        let root = self.session.join("learning");
        fs::create_dir_all(&root)?;
        let frozen = root.join("frozen");
        let data = root.join(format!("data-{cycle:04}"));
        dataset::prepare(export, &self.rehearsal, &frozen, &data)?;
        if !root.join("active.json").exists() {
            self.point(&active)?;
        }
        let Some(before) = self.scores(&active, &frozen, device)? else {
            return Ok(());
        };
        if self.initial.is_none() {
            fs::write(
                root.join("initial-scores.json"),
                serde_json::to_vec_pretty(&before)?,
            )?;
            self.initial = Some(before.clone());
        }
        let candidate = root.join(format!("candidate-{cycle:04}"));
        let mut config = checkpoint::read_config(&active)?;
        config.init_from = Some(active.clone());
        config.tokenizer = active.join("tokenizer.json");
        config.data = data;
        config.steps = self.steps;
        config.eval_every = self.steps.min(100);
        config.learning_rate = 0.00001;
        // Les exports documentaires sont du texte continu, meme apres un cours de dialogue.
        config.objective = training::Objective::NextToken;
        config.data_format = training::DataFormat::Text;
        config.warmup_steps = config.warmup_steps.min(self.steps / 10);
        if !training::train_controlled(config, &candidate, device, &|| self.stopped())? {
            return Ok(());
        }
        let Some(after) = self.scores(&candidate, &frozen, device)? else {
            return Ok(());
        };
        let promoted =
            !self.stopped() && selection::accepts(self.initial.as_ref().unwrap(), &before, &after);
        fs::write(
            root.join(format!("decision-{cycle:04}.json")),
            serde_json::to_vec_pretty(&json!({
                "before":before,"after":after,"promoted":promoted,"candidate":candidate,"previous":active,
                "test_used":false,"criterion":"web loss improves; foundation loss <= 102% of incumbent and initial baseline",
                "limitation":"prediction loss only, not a demonstration of conversational or factual competence"
            }))?,
        )?;
        if promoted {
            self.point(&candidate)?;
            self.active = Some(candidate);
            println!("Candidate retenue par les mesures de perte.");
        } else {
            println!("Candidate rejetee ; version precedente conservee.");
        }
        Ok(())
    }
    fn point(&self, run: &Path) -> Result<()> {
        let root = self.session.join("learning");
        let temp = root.join("active.next.json");
        fs::write(
            &temp,
            serde_json::to_vec_pretty(&json!({"run":fs::canonicalize(run)?}))?,
        )?;
        fs::rename(temp, root.join("active.json"))?;
        Ok(())
    }
    fn scores(&self, run: &Path, frozen: &Path, device: &Device) -> Result<Option<Scores>> {
        if self.stopped() {
            return Ok(None);
        }
        let (model, mut config) = checkpoint::load(run, device)?;
        config.objective = training::Objective::NextToken;
        config.data_format = training::DataFormat::Text;
        let tokenizer = crate::tokenization::load(&run.join("tokenizer.json"))?;
        let web = evaluation::evaluate_file(
            &model,
            &frozen.join("web.txt"),
            &tokenizer,
            &config,
            device,
        )?;
        if self.stopped() {
            return Ok(None);
        }
        let foundation = evaluation::evaluate_file(
            &model,
            &frozen.join("foundation.txt"),
            &tokenizer,
            &config,
            device,
        )?;
        Ok(Some(Scores { web, foundation }))
    }
}
