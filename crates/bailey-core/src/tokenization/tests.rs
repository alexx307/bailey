use std::fs;

use sha2::{Digest, Sha256};

use super::{ASSISTANT, EOS, USER, load, train};

const CORPUS: &str =
    "<|user|>bonjour<|assistant|>Bonjour !<|end|>\nBonjour, le monde.\nBonjour, le monde.\n";

#[test]
fn roundtrip_preserves_unseen_unicode_whitespace_and_markers() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("train.txt");
    let out = directory.path().join("tokenizer");
    fs::write(&input, CORPUS).unwrap();
    train(&input, &out, 512).unwrap();
    let tokenizer = load(&out).unwrap();
    for text in [
        "  élève, cœur, naïf 🦀🙂\t\n  ",
        "e\u{301} et é ; 中文 ; العربية",
        "<|user|> salut\n<|assistant|>Bonjour !<|end|>",
        "",
    ] {
        let encoded = tokenizer.encode(text, false).unwrap();
        assert_eq!(tokenizer.decode(encoded.get_ids(), false).unwrap(), text);
    }
    for marker in [USER, ASSISTANT, EOS] {
        let id = tokenizer.token_to_id(marker).unwrap();
        assert_eq!(tokenizer.encode(marker, false).unwrap().get_ids(), &[id]);
    }
    let from_file = load(&out.join("tokenizer.json")).unwrap();
    assert_eq!(from_file.get_vocab(true), tokenizer.get_vocab(true));
    assert!(tokenizer.get_vocab_size(true) < 512);
}

#[test]
fn existing_output_is_never_overwritten() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("train.txt");
    let out = directory.path().join("tokenizer");
    fs::write(&input, CORPUS).unwrap();
    train(&input, &out, 512).unwrap();
    let original = fs::read(out.join("tokenizer.json")).unwrap();
    fs::write(&input, "Un nouveau texte entièrement différent.").unwrap();
    assert!(train(&input, &out, 512).is_err());
    assert_eq!(fs::read(out.join("tokenizer.json")).unwrap(), original);
}

#[test]
fn only_explicit_training_file_is_read_and_fingerprinted() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("train.txt");
    let out = directory.path().join("tokenizer");
    fs::write(&input, CORPUS).unwrap();
    // Invalid UTF-8 would fail tokenizer training if either holdout were read.
    fs::write(directory.path().join("validation.txt"), [0xff, 0xfe]).unwrap();
    fs::write(directory.path().join("test.txt"), [0xff, 0xfe]).unwrap();
    let provenance = serde_json::to_vec(&serde_json::json!({
        "train_sha256":format!("{:x}", Sha256::digest(CORPUS.as_bytes())),
        "origin":"original unit fixture"
    }))
    .unwrap();
    fs::write(directory.path().join("manifest.json"), &provenance).unwrap();
    train(&input, &out, 512).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["input_sha256"],
        format!("{:x}", Sha256::digest(CORPUS.as_bytes()))
    );
    assert_eq!(manifest["input_bytes"], CORPUS.len());
    assert_eq!(manifest["requested_vocab_size"], 512);
    assert_eq!(manifest["status"], "candidate_not_frozen");
    assert_eq!(
        fs::read(out.join("source-manifest.json")).unwrap(),
        provenance
    );
    assert_eq!(
        manifest["tokenizer_sha256"],
        format!(
            "{:x}",
            Sha256::digest(fs::read(out.join("tokenizer.json")).unwrap())
        )
    );
    assert_eq!(
        manifest["actual_vocab_size"],
        load(&out).unwrap().get_vocab_size(true)
    );
    assert!(
        train(
            &directory.path().join("validation.txt"),
            &directory.path().join("forbidden"),
            512
        )
        .is_err()
    );
}

#[test]
fn invalid_input_does_not_create_an_output() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("train.txt");
    let out = directory.path().join("tokenizer");
    fs::write(&input, "").unwrap();
    assert!(train(&input, &out, 512).is_err());
    fs::write(&input, CORPUS).unwrap();
    assert!(train(&input, &out, 258).is_err());
    assert!(!out.exists());
}

#[test]
fn stale_mixture_provenance_is_rejected_before_training() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("train.txt");
    let out = directory.path().join("tokenizer");
    fs::write(&input, CORPUS).unwrap();
    fs::write(
        directory.path().join("manifest.json"),
        r#"{"train_sha256":"obsolete"}"#,
    )
    .unwrap();
    assert!(train(&input, &out, 512).is_err());
    assert!(!out.exists());
}
