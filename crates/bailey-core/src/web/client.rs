use std::time::Duration;

use anyhow::{Context, Result, ensure};
use reqwest::{Url, blocking::Client, redirect::Policy};
use serde_json::Value;

use super::{Article, body::read_bounded, parsing};

/// A fixed-host reader: no arbitrary URLs, redirects, writes, or command execution.
pub struct WikiClient {
    http: Client,
    language: String,
    endpoint: Url,
    max_response_bytes: usize,
}

impl WikiClient {
    pub fn new(language: &str, timeout_secs: u64, max_response_bytes: usize) -> Result<Self> {
        let endpoint = endpoint(language)?;
        ensure!(timeout_secs > 0, "Le délai réseau doit être positif");
        ensure!(
            max_response_bytes > 0,
            "La taille maximale doit être positive"
        );
        ensure!(
            max_response_bytes < usize::MAX,
            "Taille maximale trop grande"
        );
        let http = Client::builder()
            .https_only(true)
            .redirect(Policy::none())
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent(concat!(
                "BaileyResearch/",
                env!("CARGO_PKG_VERSION"),
                " (https://github.com/alexx307/bailey; local learning; read-only)"
            ))
            .build()
            .context("Initialisation du lecteur Wikipédia")?;
        Ok(Self {
            http,
            language: language.to_owned(),
            endpoint,
            max_response_bytes,
        })
    }

    pub fn search(&self, topic: &str, limit: usize, offset: usize) -> Result<Vec<u64>> {
        ensure!(!topic.trim().is_empty(), "Le sujet de recherche est vide");
        ensure!(
            (1..=500).contains(&limit),
            "La recherche accepte 1 à 500 résultats"
        );
        let limit = limit.to_string();
        let offset = offset.to_string();
        let value = self.query(&[
            ("list", "search"),
            ("srsearch", topic),
            ("srnamespace", "0"),
            ("srprop", ""),
            ("srlimit", &limit),
            ("sroffset", &offset),
        ])?;
        parsing::search(&value)
    }

    pub fn article(&self, id: u64) -> Result<Article> {
        ensure!(id > 0, "Identifiant de page invalide");
        let page_id = id.to_string();
        let value = self.query(&[
            ("pageids", &page_id),
            ("prop", "extracts|revisions"),
            ("explaintext", "1"),
            ("exsectionformat", "plain"),
            ("exlimit", "1"),
            ("rvprop", "ids"),
            ("rvlimit", "1"),
        ])?;
        parsing::article(&value, id, &self.language)
    }

    fn query(&self, parameters: &[(&str, &str)]) -> Result<Value> {
        let response = self
            .http
            .get(self.endpoint.clone())
            .query(&[
                ("action", "query"),
                ("format", "json"),
                ("formatversion", "2"),
                ("maxlag", "5"),
            ])
            .query(parameters)
            .send()
            .context("Connexion à Wikipédia impossible")?;
        ensure!(
            response.status().is_success(),
            "Wikipédia a répondu HTTP {}",
            response.status()
        );
        if let Some(length) = response.content_length() {
            ensure!(
                length <= self.max_response_bytes as u64,
                "Réponse Wikipédia trop grande ({length} octets)"
            );
        }
        let bytes = read_bounded(response, self.max_response_bytes)?;
        let value = serde_json::from_slice(&bytes).context("Réponse Wikipédia JSON invalide")?;
        parsing::check_error(&value)?;
        Ok(value)
    }
}

pub(super) fn endpoint(language: &str) -> Result<Url> {
    ensure!(
        (2..=12).contains(&language.len())
            && language
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'-')
            && !language.starts_with('-')
            && !language.ends_with('-'),
        "Langue invalide : utiliser 2 à 12 lettres minuscules ASCII et tirets internes"
    );
    Url::parse(&format!("https://{language}.wikipedia.org/w/api.php"))
        .context("Adresse Wikipédia invalide")
}
