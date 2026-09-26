use anyhow::{Result, bail};
use std::path::Path;

use super::{ResearchConfig, session::Session};
use crate::{
    knowledge::{Library, LibraryLock},
    web::WikiClient,
};

pub fn run(config: &ResearchConfig, library: &Path, session: &Path) -> Result<()> {
    run_with_hook(config, library, session, |_, _| Ok(()))
}

/// The hook receives an immutable dataset directory and a one-based cycle number.
/// It is called without the library lock. The hook owns its own runtime budget.
pub fn run_with_hook(
    config: &ResearchConfig,
    library: &Path,
    session: &Path,
    mut hook: impl FnMut(&Path, usize) -> Result<()>,
) -> Result<()> {
    config.validate()?;
    let reader = WikiClient::new(&config.language, 20, 2 * 1024 * 1024)?;
    {
        let _lock = LibraryLock::acquire(library)?;
        Library::open(library, config.max_library_bytes)?;
    }
    let mut session = Session::create(config, session)?;
    session.journal.write("started", 0, "Lecture Wikipedia ; memoire et corpus seulement. Les poids ne changent que via un entrainement explicite.")?;
    println!(
        "Recherche autonome : {} sujets, {} cycles maximum.",
        config.topics.len(),
        config.cycles
    );
    println!(
        "Pour arreter : creer {}",
        session.root.join("STOP").display()
    );
    let result = collect(config, library, &reader, &mut session, &mut hook);
    match &result {
        Ok(()) => session.journal.write(
            "finished",
            0,
            "Session terminee ; sources et versions conservees",
        )?,
        Err(error) => session.journal.write("failed", 0, &format!("{error:#}"))?,
    }
    result
}

fn collect(
    config: &ResearchConfig,
    library: &Path,
    reader: &WikiClient,
    session: &mut Session,
    hook: &mut impl FnMut(&Path, usize) -> Result<()>,
) -> Result<()> {
    let mut errors = 0;
    let mut has_requested = false;
    for cycle in 1..=config.cycles {
        let offset = (cycle - 1) * config.pages_per_topic;
        let mut inserted = 0;
        for topic in &config.topics {
            if !ready(session, cycle, config.interval_seconds, has_requested)? {
                return Ok(());
            }
            has_requested = true;
            println!(
                "Cycle {cycle}/{} : {:?} (offset {offset})",
                config.cycles, topic
            );
            session
                .journal
                .write("search", cycle, &format!("topic={topic:?} offset={offset}"))?;
            let ids = match reader.search(topic, config.pages_per_topic, offset) {
                Ok(ids) => ids,
                Err(error) => {
                    record_error(session, cycle, &mut errors, &error)?;
                    continue;
                }
            };
            for id in ids {
                if !ready(session, cycle, config.interval_seconds, true)? {
                    return Ok(());
                }
                let article = match reader.article(id) {
                    Ok(article) => article,
                    Err(error) => {
                        record_error(session, cycle, &mut errors, &error)?;
                        continue;
                    }
                };
                let added = {
                    let _lock = LibraryLock::acquire(library)?;
                    Library::open(library, config.max_library_bytes)?.insert(&article)?
                };
                inserted += usize::from(added);
                let event = if added { "stored" } else { "duplicate" };
                session.journal.write(
                    event,
                    cycle,
                    &format!(
                        "{} revision={} {:?}",
                        article.url, article.revision_id, article.title
                    ),
                )?;
                println!("  {event} : {:?}", article.title);
            }
        }
        if !ready(session, cycle, 0, false)? {
            return Ok(());
        }
        session.journal.write(
            "cycle_finished",
            cycle,
            &format!("{inserted} nouveaux articles"),
        )?;
        if inserted == 0 {
            continue;
        }
        let out = session
            .root
            .join("datasets")
            .join(format!("cycle-{cycle:04}"));
        let exported = {
            let _lock = LibraryLock::acquire(library)?;
            Library::open(library, config.max_library_bytes)?.export_dataset(&out, 128)
        };
        match exported {
            Ok(summary) => {
                session
                    .journal
                    .write("exported", cycle, &serde_json::to_string(&summary)?)?;
                println!(
                    "  Corpus exporte : {} ({} articles)",
                    out.display(),
                    summary.articles
                );
                if !ready(session, cycle, 0, false)? {
                    return Ok(());
                }
                hook(&out, cycle)?;
                session
                    .journal
                    .write("hook_completed", cycle, "Traitement du corpus termine")?;
            }
            Err(error) if error.to_string().starts_with("Partition ") => {
                session
                    .journal
                    .write("export_waiting", cycle, &error.to_string())?;
                println!(
                    "  Corpus incomplet : collecte a poursuivre pour les partitions reservees."
                );
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn ready(session: &mut Session, cycle: usize, seconds: u64, should_wait: bool) -> Result<bool> {
    if should_wait {
        session.wait(seconds);
    }
    if let Some(reason) = session.stop_reason() {
        session.journal.write("stopped", cycle, reason)?;
        println!("Recherche arretee : {reason}");
        return Ok(false);
    }
    Ok(true)
}

fn record_error(
    session: &mut Session,
    cycle: usize,
    count: &mut usize,
    error: &anyhow::Error,
) -> Result<()> {
    *count += 1;
    session
        .journal
        .write("network_error", cycle, &format!("{error:#}"))?;
    eprintln!("Lecture impossible ({count}/10 erreurs) : {error:#}");
    if *count >= 10 {
        bail!("Arret apres dix erreurs reseau ou articles indisponibles ; consulter events.jsonl");
    }
    Ok(())
}
