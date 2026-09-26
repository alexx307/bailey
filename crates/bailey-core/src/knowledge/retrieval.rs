use super::Library;
use crate::web::Article;
use anyhow::{Result, ensure};
use std::collections::{BTreeMap, HashSet};

const MAX_TEXT_CHARS: usize = 100_000;

/// Keep the newest saved revision of each page for retrieval and export.
pub(super) fn latest(library: &Library) -> Vec<&Article> {
    let mut pages: BTreeMap<(&str, u64), &Article> = BTreeMap::new();
    for entry in &library.entries {
        let article = &entry.article;
        let key = (article.language.as_str(), article.id);
        if pages
            .get(&key)
            .is_none_or(|saved| saved.revision_id < article.revision_id)
        {
            pages.insert(key, article);
        }
    }
    pages.into_values().collect()
}

pub fn search(library: &Library, query: &str, limit: usize) -> Result<Vec<Article>> {
    ensure!(
        query.chars().count() <= 512,
        "Recherche limitee a 512 caracteres"
    );
    ensure!(limit <= 100, "Recherche limitee a 100 resultats");
    if limit == 0 {
        return Ok(Vec::new());
    }
    let query = words(query);
    ensure!(!query.is_empty(), "La recherche doit contenir des mots");
    let mut scored = Vec::new();
    for article in latest(library) {
        let title = words(&article.title.chars().take(512).collect::<String>());
        let body = words(
            &article
                .text
                .chars()
                .take(MAX_TEXT_CHARS)
                .collect::<String>(),
        );
        let score: usize = query
            .iter()
            .map(|word| usize::from(title.contains(word)) * 6 + usize::from(body.contains(word)))
            .sum();
        if score > 0 {
            scored.push((score, article));
        }
    }
    scored.sort_by(|(score_a, article_a), (score_b, article_b)| {
        score_b.cmp(score_a).then_with(|| {
            (&article_a.title, &article_a.language, article_a.id).cmp(&(
                &article_b.title,
                &article_b.language,
                article_b.id,
            ))
        })
    });
    Ok(scored
        .into_iter()
        .take(limit)
        .map(|(_, article)| article.clone())
        .collect())
}

fn words(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect()
}
