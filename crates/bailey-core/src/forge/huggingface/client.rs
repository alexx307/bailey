use anyhow::{Result, ensure};
use reqwest::{blocking::Client, redirect::Policy};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::{io::Read, time::Duration};

pub const DATASET: &str = "epfml/FineWeb2-HQ";
pub const CONFIG: &str = "fra_Latn";

#[derive(Deserialize)]
pub struct Page<T> {
    pub rows: Vec<Row<T>>,
    pub num_rows_total: u64,
    #[serde(default)]
    pub partial: bool,
}
#[derive(Deserialize)]
pub struct Row<T> {
    pub row_idx: u64,
    pub row: T,
    #[serde(default)]
    pub truncated_cells: Vec<serde_json::Value>,
}
#[derive(Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub url: String,
    pub language: String,
    pub language_script: String,
    pub language_score: f64,
    pub quality_score: f64,
    pub date: String,
    pub dump: String,
}
/// Client generique pour l'API publique "rows" ; le dataset et sa config sont
/// fixes a la construction, jamais choisis par une entree utilisateur libre.
pub struct Reader {
    http: Client,
    dataset: &'static str,
    config: &'static str,
    pub downloaded: u64,
    max_download: u64,
}
impl Reader {
    pub fn new(dataset: &'static str, config: &'static str, max_download: u64) -> Result<Self> {
        Ok(Self {
            http: Client::builder()
                .https_only(true)
                .redirect(Policy::none())
                .timeout(Duration::from_secs(40))
                .user_agent("BaileyForge/0.1 (https://github.com/alexx307/bailey)")
                .build()?,
            dataset,
            config,
            downloaded: 0,
            max_download,
        })
    }
    pub fn page<T: DeserializeOwned>(&mut self, offset: u64, length: usize) -> Result<Page<T>> {
        ensure!((1..=100).contains(&length), "Taille de page invalide");
        let response = self
            .http
            .get("https://datasets-server.huggingface.co/rows")
            .query(&[
                ("dataset", self.dataset),
                ("config", self.config),
                ("split", "train"),
                ("offset", &offset.to_string()),
                ("length", &length.to_string()),
            ])
            .send()?;
        ensure!(
            response.status().is_success(),
            "Hugging Face HTTP {} ; aucune autre source substituee",
            response.status()
        );
        let remaining = self.max_download.saturating_sub(self.downloaded);
        ensure!(remaining > 0, "Budget de telechargement atteint");
        let limit = remaining.min(16 * 1024 * 1024);
        let mut bytes = Vec::new();
        response.take(limit + 1).read_to_end(&mut bytes)?;
        self.downloaded += bytes.len() as u64;
        ensure!(
            bytes.len() as u64 <= limit,
            "Reponse ou budget trop grand ; reduire le nombre de lignes par requete"
        );
        let page: Page<T> = serde_json::from_slice(&bytes)?;
        ensure!(!page.partial, "Apercu partiel refuse");
        ensure!(page.rows.len() <= length, "Reponse hors limite de lignes");
        for (i, row) in page.rows.iter().enumerate() {
            ensure!(
                row.row_idx == offset + i as u64,
                "Ordre de lignes inattendu"
            );
        }
        Ok(page)
    }
}
