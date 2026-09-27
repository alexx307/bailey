use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use tokenizers::AddedToken;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

use super::{ASSISTANT, EOS, USER, audit};

fn byte_tokenizer(path: &Path) {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort();
    let vocab: [(String, u32); 256] =
        std::array::from_fn(|id| (alphabet[id].to_string(), id as u32));
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    );
    let byte_level = ByteLevel::default()
        .add_prefix_space(false)
        .trim_offsets(false);
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_decoder(Some(byte_level));
    tokenizer
        .add_special_tokens([USER, ASSISTANT, EOS].map(|token| AddedToken::from(token, true)))
        .unwrap();
    tokenizer.save(path, true).unwrap();
}

fn write_probes(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

#[test]
fn multilingual_roundtrips_and_same_probe_comparison_are_exact() {
    let directory = tempfile::tempdir().unwrap();
    let tokenizer = directory.path().join("tokenizer.json");
    let probes = directory.path().join("probes.json");
    let out = directory.path().join("audit.json");
    byte_tokenizer(&tokenizer);
    let samples = json!([
        {"id":"fr", "domain":"francais", "text":"  Où est l'élève ?\t\r\n"},
        {"id":"unicode", "domain":"unicode", "text":"e\u{301} é 中文 العربية 🦀\0"},
        {"id":"code", "domain":"rust", "text":"fn carré(x: i32) -> i32 { x * x }\n"}
    ]);
    write_probes(&probes, &samples);
    audit(&tokenizer, Some(&tokenizer), &probes, &out).unwrap();
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["candidate"]["total"]["exact_roundtrips"], 3);
    assert_eq!(report["candidate"]["byte_level_alphabet"]["present"], 256);
    assert_eq!(report["candidate"]["total"]["unknown_tokens"], Value::Null);
    assert_eq!(
        report["comparison"]["total"]["candidate_over_baseline_tokens"],
        1.0
    );
    assert_eq!(report["comparison"]["vocabulary_ids_equal"], true);
    assert_eq!(
        report["candidate"]["reserved_markers"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(
        report["candidate"]["reserved_markers"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v["atomic"] == true)
    );
    for (probe, sample) in samples
        .as_array()
        .unwrap()
        .iter()
        .zip(report["candidate"]["samples"].as_array().unwrap())
    {
        let text = probe["text"].as_str().unwrap();
        assert_eq!(sample["metrics"]["bytes"], text.len());
        assert_eq!(sample["metrics"]["characters"], text.chars().count());
        assert_eq!(sample["metrics"]["tokens"], text.len());
    }
    let original = fs::read(&out).unwrap();
    assert!(audit(&tokenizer, None, &probes, &out).is_err());
    assert_eq!(original, fs::read(out).unwrap());
}

#[test]
fn missing_byte_coverage_unknown_tokens_and_lossy_roundtrip_are_reported() {
    let directory = tempfile::tempdir().unwrap();
    let tokenizer = directory.path().join("tokenizer.json");
    let probes = directory.path().join("probes.json");
    let out = directory.path().join("audit.json");
    let model = WordLevel::builder()
        .vocab([("[UNK]".into(), 0)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    Tokenizer::new(model).save(&tokenizer, true).unwrap();
    write_probes(
        &probes,
        &json!([{"id":"fr", "domain":"francais", "text":"Phrase inconnue."}]),
    );
    audit(&tokenizer, None, &probes, &out).unwrap();
    let report: Value = serde_json::from_slice(&fs::read(out).unwrap()).unwrap();
    assert_eq!(report["candidate"]["byte_level_alphabet"]["present"], 0);
    assert_eq!(report["candidate"]["total"]["unknown_tokens"], 1);
    assert_eq!(report["candidate"]["total"]["all_roundtrips_exact"], false);
    assert_eq!(
        report["candidate"]["samples"][0]["decoded_on_mismatch"],
        "[UNK]"
    );
    assert_eq!(
        report["candidate"]["reserved_markers"][0]["configured_id"],
        Value::Null
    );
}

#[test]
fn malformed_duplicate_or_reserved_probes_fail_without_creating_report() {
    let directory = tempfile::tempdir().unwrap();
    let tokenizer = directory.path().join("tokenizer.json");
    let probes = directory.path().join("probes.json");
    let out = directory.path().join("audit.json");
    byte_tokenizer(&tokenizer);
    for samples in [
        json!([]),
        json!([{"id":"a", "domain":"francais", "text":""}]),
        json!([{"id":"a", "domain":" ", "text":"Texte."}]),
        json!([{"id":"a", "domain":"francais", "text":USER}]),
        json!([{"id":"a", "domain":"francais", "text":"Texte.", "unused":true}]),
        json!([{"id":"a", "domain":"francais", "text":"Texte."}, {"id":"a", "domain":"anglais", "text":"Text."}]),
        json!([{"id":"a", "domain":"francais", "text":"Texte."}, {"id":"b", "domain":"francais", "text":"Texte."}]),
    ] {
        write_probes(&probes, &samples);
        assert!(audit(&tokenizer, None, &probes, &out).is_err());
        assert!(!out.exists());
    }
}
