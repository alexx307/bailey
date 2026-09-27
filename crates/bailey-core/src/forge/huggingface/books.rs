//! Import pilote pour PleIAs/French-PD-Books : livres francais du domaine
//! public (Gallica/BnF, auteur decede depuis 70+ ans). Schema different de
//! FineWeb2-HQ : pas d'URL, une ligne = un livre entier, score OCR fourni.
//! Le fichier README du jeu precise que la BnF restreint la reutilisation
//! commerciale des numerisations elles-memes, distinct du statut du domaine
//! public de l'oeuvre : voir https://huggingface.co/datasets/PleIAs/French-PD-Books.
use super::{client, quality};
use anyhow::{Result, ensure};
use clap::Args;
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const DATASET: &str = "PleIAs/French-PD-Books";
const CONFIG: &str = "default";

#[derive(Deserialize)]
pub struct BookDocument {
    pub file_id: String,
    pub ocr: String,
    pub title: String,
    pub date: String,
    pub author: String,
    pub page_count: i64,
    pub word_count: i64,
    pub complete_text: String,
}

#[derive(Args, serde::Serialize)]
pub struct BookImportArgs {
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 5000)]
    pub rows: usize,
    #[arg(long, default_value_t = 0)]
    pub offset: u64,
    #[arg(long, default_value_t = 5)]
    pub page_size: usize,
    #[arg(long, default_value_t = 400)]
    pub max_text_mb: u64,
    #[arg(long, default_value_t = 1024)]
    pub max_download_mb: u64,
    #[arg(long, default_value_t = 45)]
    pub max_minutes: u64,
    #[arg(long, default_value_t = 4)]
    pub interval_seconds: u64,
    #[arg(long, default_value_t = 60)]
    pub min_ocr_score: u32,
}

fn reason(doc: &BookDocument, min_ocr_score: u32) -> Option<&'static str> {
    let ocr_score: f64 = match doc.ocr.trim().parse() {
        Ok(value) => value,
        Err(_) => return Some("ocr_field_invalid"),
    };
    if !(0.0..=100.0).contains(&ocr_score) || ocr_score < f64::from(min_ocr_score) {
        return Some("low_ocr_score");
    }
    if doc.word_count < 500 {
        return Some("too_few_words");
    }
    let chars = doc.complete_text.chars().count();
    if !(2_000..=3_000_000).contains(&chars) {
        return Some("length");
    }
    if doc
        .complete_text
        .chars()
        .any(|c| c == '\u{fffd}' || (c.is_control() && !c.is_whitespace()))
    {
        return Some("corrupted_characters");
    }
    if [
        "<script",
        "<html",
        crate::tokenization::USER,
        crate::tokenization::ASSISTANT,
        crate::tokenization::EOS,
    ]
    .iter()
    .any(|s| doc.complete_text.contains(s))
    {
        return Some("markup_or_reserved_tokens");
    }
    None
}

pub fn import(args: &BookImportArgs) -> Result<()> {
    ensure!(
        (1..=25).contains(&args.page_size),
        "Taille de page attendue : 1..25 (lignes = livres entiers)"
    );
    ensure!(!args.out.exists(), "Utiliser un nouveau dossier d'import");
    ensure!(
        (1..=5000).contains(&args.rows) && args.offset.checked_add(args.rows as u64).is_some(),
        "Import pilote : 1..5000 lignes et offset valide"
    );
    ensure!(
        (1..=400).contains(&args.max_text_mb)
            && (1..=1024).contains(&args.max_download_mb)
            && (1..=60).contains(&args.max_minutes)
            && (1..=60).contains(&args.interval_seconds),
        "Budgets hors limites du pilote"
    );
    ensure!(
        args.min_ocr_score <= 100,
        "Score OCR minimal attendu : 0..100"
    );
    let mut reader =
        client::Reader::new(DATASET, CONFIG, true, args.max_download_mb * 1024 * 1024)?;
    fs::create_dir_all(args.out.parent().unwrap_or(std::path::Path::new(".")))?;
    fs::create_dir(&args.out)?;
    let mut files = Vec::new();
    for name in ["train", "validation", "test"] {
        files.push(File::create(args.out.join(format!("{name}.txt")))?);
    }
    let mut rejected = BTreeMap::<String, usize>::new();
    let mut counts = [0usize; 3];
    let mut sizes = [0usize; 3];
    let mut dedup = quality::Dedup::default();
    let mut sources = Vec::new();
    let mut visited = 0usize;
    let mut import_reason = "row_limit";
    let started = Instant::now();
    let fetched = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let result = (|| -> Result<()> {
        'pages: while visited < args.rows {
            if args.out.join("STOP").exists() {
                import_reason = "STOP";
                break;
            }
            if started.elapsed() >= Duration::from_secs(args.max_minutes * 60) {
                import_reason = "time_budget";
                break;
            }
            let page = reader.page::<BookDocument>(
                args.offset + visited as u64,
                (args.rows - visited).min(args.page_size),
            )?;
            if page.rows.is_empty() {
                import_reason = "end_of_source";
                break;
            }
            let total_rows = page.num_rows_total;
            for mut row in page.rows {
                visited += 1;
                row.row.complete_text = row
                    .row
                    .complete_text
                    .replace("\r\n", "\n")
                    .trim()
                    .to_owned();
                let failure = if !row.truncated_cells.is_empty() {
                    Some("truncated_row")
                } else {
                    reason(&row.row, args.min_ocr_score)
                };
                if let Some(failure) = failure {
                    *rejected.entry(failure.into()).or_default() += 1;
                    continue;
                }
                let split = quality::partition(&row.row.file_id);
                let bytes = row.row.complete_text.len() + crate::tokenization::EOS.len() + 1;
                if sizes.iter().sum::<usize>() as u64 + bytes as u64
                    > args.max_text_mb * 1024 * 1024
                {
                    import_reason = "text_budget";
                    break 'pages;
                }
                if let Err(why) = dedup.accept(&row.row.complete_text, &row.row.file_id, split) {
                    *rejected.entry(why.into()).or_default() += 1;
                    continue;
                }
                writeln!(
                    files[split],
                    "{}{}",
                    row.row.complete_text,
                    crate::tokenization::EOS
                )?;
                counts[split] += 1;
                sizes[split] += bytes;
                sources.push(json!({
                    "file_id": row.row.file_id, "row_index": row.row_idx,
                    "title": row.row.title, "author": row.row.author, "date": row.row.date,
                    "ocr": row.row.ocr, "page_count": row.row.page_count,
                    "word_count": row.row.word_count,
                    "text_sha256": quality::hash(&row.row.complete_text),
                    "partition": (["train", "validation", "test"][split]),
                }));
            }
            if visited.is_multiple_of(50) {
                println!(
                    "HF livres : {visited} lignes examinees, {} retenues, {} octets texte",
                    counts.iter().sum::<usize>(),
                    sizes.iter().sum::<usize>()
                );
            }
            if args.offset + visited as u64 >= total_rows {
                import_reason = "end_of_source";
                break;
            }
            std::thread::sleep(Duration::from_secs(args.interval_seconds));
        }
        Ok(())
    })();
    for file in &mut files {
        file.flush()?;
        file.sync_all()?;
    }
    let error = result.as_ref().err().map(|e| format!("{e:#}"));
    let complete = result.is_ok()
        && counts.iter().all(|&n| n > 0)
        && import_reason != "STOP"
        && import_reason != "time_budget";
    let manifest = json!({
        "dataset": DATASET, "config": CONFIG, "remote_split": "train",
        "local_splits": ["train", "validation", "test"],
        "counts": counts, "text_bytes": sizes, "rows_visited": visited,
        "downloaded_body_bytes": reader.downloaded,
        "source_reported_partial_index": reader.saw_partial,
        "dataset_license": "Public domain (EU rule: author deceased 70+ years) per the dataset curator's own claim; no machine-readable SPDX license tag on the dataset card",
        "license_note": "Digitizations are sourced from Gallica (BnF). The BnF's own terms of use restrict commercial reuse of the digitized files themselves, separately from the public-domain status of the underlying works. Not verified independently per book.",
        "license_url": "https://huggingface.co/datasets/PleIAs/French-PD-Books",
        "source_card": "https://huggingface.co/datasets/PleIAs/French-PD-Books",
        "fetched_at_unix": fetched, "settings": args, "complete": complete,
        "stop_reason": import_reason, "error": error, "rejected": rejected,
        "sources": sources,
        "policy": "min OCR score filter; word/character length checks; exact text dedup; word-trigram SimHash Hamming<=3; significant shared paragraphs rejected across splits; SHA256(file_id) split",
        "limitations": [
            "Full books (often historical, pre-1950s French) are kept as single documents, not web-snippet sized.",
            "Historical/older French register: not representative of contemporary conversational French; kept as a separate source, not merged into the main pretraining mix by default.",
            "OCR score filter is a heuristic on a self-reported field; residual OCR noise (misrecognized characters, broken layout, tables) is possible even above the threshold.",
            "No independent language verification: relies on the dataset's own French-only curation.",
            "Rows API does not pin a dataset revision; exported local snapshot is fingerprinted by Forge.",
            "This bounded pilot samples a tiny fraction of the 289,000-book collection; it is not a representative literary corpus.",
            "The dataset-server preview reported a partial index for this dataset (num_rows_total reflects only the indexed prefix, not the full 289,000 books); pagination cannot reach the rest of the collection through this API.",
        ]
    });
    let name = if complete {
        "manifest.json"
    } else {
        "incomplete-manifest.json"
    };
    fs::write(args.out.join(name), serde_json::to_vec_pretty(&manifest)?)?;
    result?;
    ensure!(
        complete,
        "Import incomplet : consulter incomplete-manifest.json ; ne pas entrainer dessus"
    );
    println!(
        "Import termine : {} documents ; train/validation/test {:?} ; {} octets texte",
        counts.iter().sum::<usize>(),
        counts,
        sizes.iter().sum::<usize>()
    );
    Ok(())
}
