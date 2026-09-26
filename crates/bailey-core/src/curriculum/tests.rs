use std::collections::HashSet;
use std::fs;

use super::{
    dataset::{examples, split},
    prepare,
};

#[test]
fn complete_units_have_distinct_ids_and_do_not_cross_partitions() {
    let units = examples();
    assert_eq!(units.len(), 120);
    let mut ids = HashSet::new();
    let mut texts = HashSet::new();
    let mut partitions: [HashSet<String>; 3] = Default::default();
    for unit in units {
        assert!(ids.insert(unit.id.clone()));
        assert!(texts.insert(unit.text.clone()));
        partitions[split(&unit.id)].insert(unit.text);
    }
    assert!(partitions.iter().all(|part| !part.is_empty()));
    for left in 0..3 {
        for right in (left + 1)..3 {
            assert!(partitions[left].is_disjoint(&partitions[right]));
        }
    }
}

#[test]
fn preparation_preserves_existing_corpora_and_explains_its_limits() {
    let directory = tempfile::tempdir().unwrap();
    let out = directory.path().join("seed");
    prepare(&out).unwrap();
    let original = fs::read(out.join("train.txt")).unwrap();
    assert!(prepare(&out).is_err());
    assert_eq!(original, fs::read(out.join("train.txt")).unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["purpose"], "pipeline_smoke_test");
    assert_eq!(manifest["not_general_language_pretraining"], true);
    for (index, name) in ["train", "validation", "test"].into_iter().enumerate() {
        let text = fs::read_to_string(out.join(format!("{name}.txt"))).unwrap();
        let expected: String = examples()
            .into_iter()
            .filter(|unit| split(&unit.id) == index)
            .map(|unit| unit.text)
            .collect();
        assert_eq!(text, expected);
        assert_eq!(manifest["partitions"][index]["bytes"], text.len());
    }
}
