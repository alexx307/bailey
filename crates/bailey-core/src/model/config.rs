use candle_core::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            hidden_size: 768,
            intermediate_size: 2_048,
            num_layers: 12,
            num_attention_heads: 12,
            num_kv_heads: 4,
            max_seq_len: 1_024,
            rope_theta: 10_000.,
            rms_norm_eps: 1e-5,
        }
    }
}

impl CoreConfig {
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 128,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if [
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_layers,
            self.num_attention_heads,
            self.num_kv_heads,
            self.max_seq_len,
        ]
        .contains(&0)
        {
            bail!("Toutes les dimensions du modèle doivent être positives");
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            bail!("hidden_size doit être divisible par num_attention_heads");
        }
        if !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            bail!("num_attention_heads doit être divisible par num_kv_heads");
        }
        if !(self.hidden_size / self.num_attention_heads).is_multiple_of(2) {
            bail!("RoPE nécessite une dimension paire par tête d'attention");
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0. {
            bail!("rope_theta doit être fini et strictement positif");
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0. {
            bail!("rms_norm_eps doit être fini et strictement positif");
        }
        Ok(())
    }

    /// Nombre de scalaires appris. La sortie réutilise exactement l'embedding.
    pub fn parameter_count(&self) -> Result<usize> {
        self.validate()?;
        let hidden = self.hidden_size as u128;
        let kv = (self.hidden_size / self.num_attention_heads * self.num_kv_heads) as u128;
        let embedding = (self.vocab_size as u128).checked_mul(hidden);
        let attention = hidden
            .checked_mul(hidden)
            .and_then(|x| x.checked_add(hidden.checked_mul(kv)?))
            .and_then(|x| x.checked_mul(2));
        let feed_forward = hidden
            .checked_mul(self.intermediate_size as u128)
            .and_then(|x| x.checked_mul(3));
        let total = attention
            .and_then(|x| x.checked_add(feed_forward?))
            .and_then(|x| x.checked_add(hidden.checked_mul(2)?))
            .and_then(|x| x.checked_mul(self.num_layers as u128))
            .and_then(|x| x.checked_add(embedding?))
            .and_then(|x| x.checked_add(hidden));
        match total.and_then(|value| usize::try_from(value).ok()) {
            Some(value) => Ok(value),
            None => bail!("Le nombre de paramètres dépasse la taille représentable"),
        }
    }
}
