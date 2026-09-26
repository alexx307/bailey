use super::{Library, StoredArticle, hashing};
use crate::web::Article;
use anyhow::{Context, Result, ensure};
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

pub fn open(root: &Path, max_bytes: u64) -> Result<Library> {
    ensure!(
        max_bytes > 0,
        "Le budget de la bibliotheque doit etre positif"
    );
    fs::create_dir_all(root.join("articles"))?;
    ensure!(
        used_bytes(root)? <= max_bytes,
        "La bibliotheque depasse deja le budget de {max_bytes} octets"
    );
    let mut entries = Vec::new();
    let mut revisions = HashSet::new();
    let mut contents = HashSet::new();
    for path in files(&root.join("articles"))? {
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let entry: StoredArticle = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("Article illisible : {}", path.display()))?;
        validate(&entry.article)?;
        ensure!(entry.schema_version == 1, "Version d'article inconnue");
        ensure!(
            entry.identity_sha256 == hashing::identity(&entry.article)
                && entry.text_sha256 == hashing::sha256(&entry.article.text),
            "Integrite invalide : {}",
            path.display()
        );
        ensure!(
            path == article_path(root, &entry),
            "Chemin d'article incoherent : {}",
            path.display()
        );
        ensure!(
            revisions.insert((entry.identity_sha256.clone(), entry.article.revision_id))
                && contents.insert(entry.text_sha256.clone()),
            "Doublon dans la bibliotheque : {}",
            path.display()
        );
        entries.push(entry);
    }
    Ok(Library {
        root: root.to_owned(),
        max_bytes,
        entries,
    })
}

pub fn insert(library: &mut Library, article: &Article) -> Result<bool> {
    validate(article)?;
    let entry = StoredArticle {
        schema_version: 1,
        identity_sha256: hashing::identity(article),
        text_sha256: hashing::sha256(&article.text),
        article: article.clone(),
    };
    if library.entries.iter().any(|saved| {
        saved.text_sha256 == entry.text_sha256
            || (saved.identity_sha256 == entry.identity_sha256
                && saved.article.revision_id == article.revision_id)
    }) {
        return Ok(false);
    }
    let bytes = serde_json::to_vec_pretty(&entry)?;
    let total = used_bytes(&library.root)?.checked_add(bytes.len() as u64);
    ensure!(
        total.is_some_and(|total| total <= library.max_bytes),
        "Budget de bibliotheque atteint ({} octets) ; augmentez --max-library-mb",
        library.max_bytes
    );
    let path = article_path(&library.root, &entry);
    fs::create_dir_all(path.parent().expect("article has a parent"))?;
    write_new(&path, &bytes)?;
    library.entries.push(entry);
    Ok(true)
}

fn validate(article: &Article) -> Result<()> {
    ensure!(
        article.id > 0 && article.revision_id > 0,
        "Identifiants d'article invalides"
    );
    ensure!(
        !article.language.is_empty()
            && article
                .language
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'-'),
        "Langue d'article invalide"
    );
    ensure!(!article.title.trim().is_empty(), "Titre d'article vide");
    ensure!(!article.text.trim().is_empty(), "Texte d'article vide");
    ensure!(!article.url.trim().is_empty(), "URL d'article manquante");
    ensure!(
        !article.license.trim().is_empty(),
        "Licence d'article manquante"
    );
    Ok(())
}

fn article_path(root: &Path, entry: &StoredArticle) -> PathBuf {
    root.join("articles")
        .join(&entry.identity_sha256)
        .join(format!(
            "{}-{}.json",
            entry.article.revision_id, entry.text_sha256
        ))
}

pub(super) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = path.with_file_name(format!(
        ".pending-{}-{timestamp}-{sequence}.tmp",
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("Creation temporaire : {}", temporary.display()))?;
    let written = file.write_all(bytes).and_then(|()| file.sync_all());
    drop(file);
    // Linking publishes a complete file atomically and fails if the name exists.
    // Both paths share one directory/filesystem; no existing file is overwritten.
    let result = written.and_then(|()| fs::hard_link(&temporary, path));
    let cleanup = fs::remove_file(&temporary);
    result
        .and(cleanup)
        .with_context(|| format!("Publication sans ecrasement : {}", path.display()))
}

fn used_bytes(root: &Path) -> Result<u64> {
    files(root)?.into_iter().try_fold(0_u64, |total, path| {
        total
            .checked_add(fs::metadata(path)?.len())
            .context("Taille de bibliotheque trop grande")
    })
}

fn files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut pending = vec![root.to_owned()];
    let mut found = Vec::new();
    while let Some(directory) = pending.pop() {
        ensure!(
            !fs::symlink_metadata(&directory)?.file_type().is_symlink(),
            "Lien symbolique non pris en charge : {}",
            directory.display()
        );
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            ensure!(!kind.is_symlink(), "Lien symbolique non pris en charge");
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() {
                found.push(entry.path());
            }
        }
    }
    found.sort();
    Ok(found)
}
