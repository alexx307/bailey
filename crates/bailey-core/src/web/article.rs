use serde::{Deserialize, Serialize};

/// Source metadata remains attached to every downloaded text for attribution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Article {
    pub id: u64,
    pub revision_id: u64,
    pub title: String,
    pub url: String,
    pub language: String,
    pub text: String,
    pub license: String,
    pub fetched_at_unix: u64,
}

pub(super) const LICENSE: &str = "CC BY-SA 4.0; \
    https://creativecommons.org/licenses/by-sa/4.0/; \
    additional terms may apply: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use";
