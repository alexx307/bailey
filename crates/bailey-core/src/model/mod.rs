//! Cœur causal dense : poids partagés, RMSNorm, RoPE, GQA et SwiGLU.
//! Les opérations restent différentiables pour l'entraînement avec Candle.

mod attention;
mod block;
mod config;
mod feed_forward;
mod network;
mod normalization;
mod rotary;

pub use config::CoreConfig;
pub use network::CoreModel;

#[cfg(test)]
mod tests;
