use super::*;
use serde_json::Value;

fn fixture(root: &Path) -> std::path::PathBuf {
    let provenance = root.join("source.json");
    fs::write(&provenance, r#"{"origin":"original test fixtures"}"#).unwrap();
    let texts = [
        "Français : été et lumière. ".repeat(400),
        "English prose and spaces. ".repeat(100),
        "fn main() { let x = 2; }\n".repeat(60),
    ];
    let sources = texts.iter().zip(DOMAINS).enumerate().map(|(index, (text, domain))| {
        let name = format!("input-{index}.txt");
        fs::write(root.join(&name), text).unwrap();
        json!({"id":name,"file":name,"domain":domain,"partition":"train","sha256":hash(text.as_bytes()),"provenance":"source.json"})
    }).collect::<Vec<_>>();
    let config = root.join("config.json");
    fs::write(
        &config,
        serde_json::to_vec(
            &json!({"seed":42,"max_text_bytes":100000,"percentages":[70,15,15],"sources":sources}),
        )
        .unwrap(),
    )
    .unwrap();
    config
}

#[test]
fn deterministic_mixture_is_bounded_balanced_and_does_not_repeat_fragments() {
    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    let a = root.path().join("a");
    let b = root.path().join("b");
    prepare(&config, &a).unwrap();
    prepare(&config, &b).unwrap();
    let train = fs::read(a.join("train.txt")).unwrap();
    assert_eq!(train, fs::read(b.join("train.txt")).unwrap());
    let manifest: Value =
        serde_json::from_slice(&fs::read(a.join("manifest.json")).unwrap()).unwrap();
    let mut ranges = HashSet::new();
    for fragment in manifest["fragments"].as_array().unwrap() {
        assert!(ranges.insert((
            fragment["source_index"].as_u64().unwrap(),
            fragment["offset"].as_u64().unwrap()
        )));
    }
    for (domain, target) in manifest["domains"]
        .as_array()
        .unwrap()
        .iter()
        .zip([70., 15., 15.])
    {
        assert!(
            domain["retained_bytes"].as_u64().unwrap()
                <= domain["available_bytes"].as_u64().unwrap()
        );
        assert!((domain["actual_source_percent"].as_f64().unwrap() - target).abs() < 0.1);
    }
    assert!(prepare(&config, &a).is_err());
    assert_eq!(train, fs::read(a.join("train.txt")).unwrap());
}

#[test]
fn invalid_partitions_and_changed_sources_are_rejected_before_output() {
    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    let mut value: Value = serde_json::from_slice(&fs::read(&config).unwrap()).unwrap();
    value["sources"][0]["partition"] = json!("test");
    fs::write(&config, serde_json::to_vec(&value).unwrap()).unwrap();
    let out = root.path().join("out");
    assert!(prepare(&config, &out).is_err());
    assert!(!out.exists());
    value["sources"][0]["partition"] = json!("train");
    fs::write(&config, serde_json::to_vec(&value).unwrap()).unwrap();
    fs::write(root.path().join("input-0.txt"), "Texte modifié").unwrap();
    assert!(prepare(&config, &out).is_err());
    assert!(!out.exists());
}

#[test]
fn incomplete_import_cannot_be_laundered_through_a_mixture() {
    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    fs::write(
        root.path().join("source.json"),
        r#"{"complete":false,"error":"HTTP 429"}"#,
    )
    .unwrap();
    let out = root.path().join("out");
    assert!(
        prepare(&config, &out)
            .unwrap_err()
            .to_string()
            .contains("incomplete")
    );
    assert!(!out.exists());
}

#[test]
fn output_budget_includes_fragment_separators() {
    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    let mut value: Value = serde_json::from_slice(&fs::read(&config).unwrap()).unwrap();
    value["max_text_bytes"] = json!(1024);
    fs::write(&config, serde_json::to_vec(&value).unwrap()).unwrap();
    let out = root.path().join("out");
    prepare(&config, &out).unwrap();
    assert!(fs::metadata(out.join("train.txt")).unwrap().len() <= 1024);
}
