use super::*;
use anyhow::Result;
use std::fs;

#[test]
fn roundtrip_detects_tampering_and_train_never_reads_test_shards() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let data = temp.path().join("data");
    fs::create_dir(&data)?;
    fs::write(
        data.join("manifest.json"),
        r#"{"origin":"original test fixtures"}"#,
    )?;
    for (name, text) in [
        ("train", "Bonjour. Le chat dort dans la maison."),
        ("validation", "Le jardin est vert."),
        ("test", "La lune brille ce soir."),
    ] {
        fs::write(data.join(format!("{name}.txt")), text)?;
    }
    let tokens = temp.path().join("tokens");
    crate::tokenization::train(&data.join("train.txt"), &tokens, 300)?;
    let out = temp.path().join("shards");
    let manifest = build(&data, &tokens, &out, 8)?;
    let mut escaped = manifest.clone();
    escaped.partitions[0].shards[0].artifact.file = "../outside.u32".into();
    fs::write(
        out.join("forge-manifest.json"),
        serde_json::to_vec(&escaped)?,
    )?;
    assert!(read_manifest(&out).is_err());
    fs::write(
        out.join("forge-manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    let tokenizer = crate::tokenization::load(&tokens)?;
    let expected = tokenizer
        .encode(fs::read_to_string(data.join("train.txt"))?, false)
        .unwrap();
    assert_eq!(
        load_partition(&out, "train", &tokens, 300)?,
        expected.get_ids()
    );
    assert!(build(&data, &tokens, &out, 8).is_err());
    let test_file = out.join(&manifest.partitions[2].shards[0].artifact.file);
    fs::write(test_file, b"broken")?;
    assert!(load_partition(&out, "train", &tokens, 300).is_ok());
    assert!(load_partition(&out, "test", &tokens, 300).is_err());
    fs::write(
        out.join(&manifest.partitions[0].shards[0].artifact.file),
        b"broken",
    )?;
    assert!(load_partition(&out, "train", &tokens, 300).is_err());
    Ok(())
}

#[test]
fn rejects_cross_partition_fragments_and_path_traversal() -> Result<()> {
    let mut detector = duplicate::Detector::default();
    let repeated = "Une phrase assez longue pour constituer un passage significatif avec plus de huit mots distincts.";
    detector.check("train", repeated)?;
    assert!(
        detector
            .check("validation", &format!("Titre\n{repeated}"))
            .is_err()
    );
    assert!(super::manifest::shard_name("train", 0).starts_with("train/"));
    Ok(())
}
