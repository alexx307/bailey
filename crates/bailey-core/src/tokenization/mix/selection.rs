use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct Fragment {
    pub source_index: usize,
    pub offset: usize,
    pub bytes: usize,
}

/// Frontieres UTF-8, sans nettoyage destructif du code ou des espaces.
pub fn fragments(source_index: usize, text: &str) -> Vec<Fragment> {
    let mut offset = 0;
    let mut result = Vec::new();
    while offset < text.len() {
        let mut end = (offset + 4096).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        result.push(Fragment {
            source_index,
            offset,
            bytes: end - offset,
        });
        offset = end;
    }
    result
}

pub fn select(
    mut pool: Vec<Fragment>,
    texts: &[String],
    budget: usize,
    seed: u64,
) -> Vec<Fragment> {
    pool.shuffle(&mut StdRng::seed_from_u64(seed));
    let mut remaining = budget;
    let mut selected = Vec::new();
    for mut item in pool {
        let text = &texts[item.source_index];
        item.bytes = item.bytes.min(remaining);
        while !text.is_char_boundary(item.offset + item.bytes) {
            item.bytes -= 1;
        }
        if item.bytes == 0 {
            continue;
        }
        remaining -= item.bytes;
        selected.push(item);
        if remaining == 0 {
            break;
        }
    }
    selected
}
