//! Import pilote borne via l'API publique des lignes. Aucun poids externe.
mod client;
mod quality;
use anyhow::{Result, ensure};
use clap::Args;
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Args, serde::Serialize)]
pub struct ImportArgs {
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 500)]
    pub rows: usize,
    #[arg(long, default_value_t = 0)]
    pub offset: u64,
    #[arg(long, default_value_t = 25)]
    pub page_size: usize,
    #[arg(long, default_value_t = 20)]
    pub max_text_mb: u64,
    #[arg(long, default_value_t = 128)]
    pub max_download_mb: u64,
    #[arg(long, default_value_t = 10)]
    pub max_minutes: u64,
}

pub fn import(args: &ImportArgs) -> Result<()> {
    ensure!(
        (1..=100).contains(&args.page_size),
        "Taille de page attendue : 1..100"
    );
    ensure!(!args.out.exists(), "Utiliser un nouveau dossier d'import");
    ensure!(
        (1..=5000).contains(&args.rows) && args.offset.checked_add(args.rows as u64).is_some(),
        "Import pilote : 1..5000 lignes et offset valide"
    );
    ensure!(
        (1..=200).contains(&args.max_text_mb)
            && (1..=1024).contains(&args.max_download_mb)
            && (1..=60).contains(&args.max_minutes),
        "Budgets hors limites du pilote"
    );
    let mut reader = client::Reader::new(args.max_download_mb * 1024 * 1024)?;
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
    let mut reason = "row_limit";
    let started = Instant::now();
    let fetched = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let result = (|| -> Result<()> {
        'pages: while visited < args.rows {
            if args.out.join("STOP").exists() {
                reason = "STOP";
                break;
            }
            if started.elapsed() >= Duration::from_secs(args.max_minutes * 60) {
                reason = "time_budget";
                break;
            }
            let page = reader.page(
                args.offset + visited as u64,
                (args.rows - visited).min(args.page_size),
            )?;
            if page.rows.is_empty() {
                reason = "end_of_source";
                break;
            }
            let total_rows = page.num_rows_total;
            for mut row in page.rows {
                visited += 1;
                row.row.text = row.row.text.replace("\r\n", "\n").trim().to_owned();
                let failure = if !row.truncated_cells.is_empty() {
                    Some("truncated_row")
                } else {
                    quality::reason(&row.row)
                };
                if let Some(failure) = failure {
                    *rejected.entry(failure.into()).or_default() += 1;
                    continue;
                }
                let mut url = match reqwest::Url::parse(&row.row.url) {
                    Ok(url)
                        if matches!(url.scheme(), "http" | "https") && url.host_str().is_some() =>
                    {
                        url
                    }
                    _ => {
                        *rejected.entry("invalid_url".into()).or_default() += 1;
                        continue;
                    }
                };
                url.set_fragment(None);
                let host = url.host_str().unwrap().to_lowercase();
                let split = quality::partition(&host);
                let bytes = row.row.text.len() + crate::tokenization::EOS.len() + 1;
                if sizes.iter().sum::<usize>() as u64 + bytes as u64
                    > args.max_text_mb * 1024 * 1024
                {
                    reason = "text_budget";
                    break 'pages;
                }
                if let Err(why) = dedup.accept(&row.row.text, url.as_str(), split) {
                    *rejected.entry(why.into()).or_default() += 1;
                    continue;
                }
                writeln!(files[split], "{}{}", row.row.text, crate::tokenization::EOS)?;
                counts[split] += 1;
                sizes[split] += bytes;
                sources.push(json!({"id":row.row.id,"row_index":row.row_idx,"url":url.as_str(),"host":host,"date":row.row.date,"dump":row.row.dump,"language_score":row.row.language_score,"quality_score":row.row.quality_score,"text_sha256":quality::hash(&row.row.text),"partition":(["train","validation","test"][split])}));
            }
            if visited.is_multiple_of(50) {
                println!(
                    "HF : {visited} lignes examinees, {} retenues, {} octets texte",
                    counts.iter().sum::<usize>(),
                    sizes.iter().sum::<usize>()
                );
            }
            if args.offset + visited as u64 >= total_rows {
                reason = "end_of_source";
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
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
        && reason != "STOP"
        && reason != "time_budget";
    let manifest = json!({"dataset":client::DATASET,"config":client::CONFIG,"remote_split":"train","local_splits":["train","validation","test"],"counts":counts,"text_bytes":sizes,"rows_visited":visited,"downloaded_body_bytes":reader.downloaded,"dataset_license":"ODC-By-1.0","license_url":"https://opendatacommons.org/licenses/by/1-0/","source_terms":"https://commoncrawl.org/terms-of-use","source_card":"https://huggingface.co/datasets/epfml/FineWeb2-HQ","fetched_at_unix":fetched,"settings":args,"complete":complete,"stop_reason":reason,"error":error,"rejected":rejected,"sources":sources,"policy":"fixed French subset; length/characters/language score checks; exact URL/text dedup; word-trigram SimHash Hamming<=3; significant shared paragraphs rejected across splits; SHA256(host) split","limitations":["Dataset license does not replace source content rights; retain original URLs","No independent language/factual verifier; SimHash is heuristic","Sequential API sample is not a representative large training corpus","Rows API does not pin a dataset revision; exported local snapshot is fingerprinted by Forge","API may transfer embedding columns, but they are discarded and never stored or used to train Bailey","This bounded pilot is not a multi-gigabyte Parquet streaming importer"]});
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
        "Import termine : {} documents ; train/validation/test {:?}",
        counts.iter().sum::<usize>(),
        counts
    );
    Ok(())
}
