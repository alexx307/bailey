use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use super::{Article, article::LICENSE};

pub(super) fn check_error(value: &Value) -> Result<()> {
    if let Some(error) = value.get("error") {
        let code = error
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or("inconnue");
        let info = error
            .get("info")
            .and_then(Value::as_str)
            .unwrap_or("sans détail");
        bail!("Erreur de l'API Wikipédia ({code}) : {info}");
    }
    ensure!(
        value.get("errors").is_none(),
        "L'API Wikipédia a retourné des erreurs"
    );
    Ok(())
}

pub(super) fn search(value: &Value) -> Result<Vec<u64>> {
    check_error(value)?;
    let matches = value
        .pointer("/query/search")
        .and_then(Value::as_array)
        .context("Liste des résultats Wikipédia absente")?;
    matches
        .iter()
        .map(|item| {
            ensure!(
                item.get("ns").and_then(Value::as_u64) == Some(0),
                "Résultat hors espace encyclopédique"
            );
            item.get("pageid")
                .and_then(Value::as_u64)
                .filter(|id| *id > 0)
                .context("Identifiant absent du résultat Wikipédia")
        })
        .collect()
}

pub(super) fn article(value: &Value, id: u64, language: &str) -> Result<Article> {
    check_error(value)?;
    let pages = value
        .pointer("/query/pages")
        .and_then(Value::as_array)
        .context("Pages absentes de la réponse Wikipédia")?;
    let page = pages
        .iter()
        .find(|page| page.get("pageid").and_then(Value::as_u64) == Some(id))
        .context("Article absent ou supprimé")?;
    ensure!(page.get("missing").is_none(), "Article Wikipédia manquant");
    ensure!(
        page.get("ns").and_then(Value::as_u64) == Some(0),
        "Page hors espace encyclopédique"
    );
    let title = page
        .get("title")
        .and_then(Value::as_str)
        .filter(|title| !title.trim().is_empty())
        .context("Titre de l'article absent")?;
    let text = page
        .get("extract")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .context("Aucun extrait texte disponible pour cet article")?;
    let revision_id = page
        .pointer("/revisions/0/revid")
        .and_then(Value::as_u64)
        .filter(|revision| *revision > 0)
        .context("Révision de l'article absente")?;
    let fetched_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("Horloge du système antérieure à 1970")?
        .as_secs();
    Ok(Article {
        id,
        revision_id,
        title: title.to_owned(),
        url: format!("https://{language}.wikipedia.org/?curid={id}"),
        language: language.to_owned(),
        text: text.to_owned(),
        license: LICENSE.to_owned(),
        fetched_at_unix,
    })
}
