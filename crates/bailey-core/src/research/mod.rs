//! Bounded online reading and dataset preparation. This module never trains weights.

mod config;
mod journal;
mod manual;
mod runner;
mod session;

pub use config::ResearchConfig;
pub use manual::{lookup, search_and_store};
pub use runner::{run, run_with_hook};

#[cfg(test)]
mod tests;
