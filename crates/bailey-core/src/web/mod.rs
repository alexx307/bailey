//! Read-only access to public encyclopedia articles, never executable instructions.

mod article;
mod body;
mod client;
mod parsing;

pub use article::Article;
pub use client::WikiClient;

#[cfg(test)]
mod tests;
