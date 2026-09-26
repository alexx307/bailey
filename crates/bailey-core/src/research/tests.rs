use std::{fs, time::Instant};

use super::{ResearchConfig, session::Session};

#[test]
fn config_is_bounded_but_does_not_restrict_subject_matter() {
    let mut config = ResearchConfig {
        topics: vec!["un sujet nouveau quelconque".into()],
        ..Default::default()
    };
    assert!(config.validate().is_ok());
    config.cycles = 0;
    assert!(config.validate().is_err());
    config.cycles = 1;
    config.max_minutes = u64::MAX;
    assert!(config.validate().is_err());
    config.max_minutes = 1;
    config.interval_seconds = 0;
    assert!(config.validate().is_err());
}

#[test]
fn stop_file_interrupts_delay_and_sessions_never_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session");
    let config = ResearchConfig::default();
    let session = Session::create(&config, &path).unwrap();
    assert!(Session::create(&config, &path).is_err());
    fs::write(path.join("STOP"), "").unwrap();
    let start = Instant::now();
    assert!(!session.wait(60));
    assert!(start.elapsed().as_secs() < 1);
    assert_eq!(session.stop_reason(), Some("Fichier STOP detecte"));
}

#[test]
fn journal_retains_control_characters_as_json_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session");
    let mut session = Session::create(&ResearchConfig::default(), &path).unwrap();
    session
        .journal
        .write("stored", 1, "Titre\nmessage\u{1b}[0m")
        .unwrap();
    let text = fs::read_to_string(path.join("events.jsonl")).unwrap();
    assert_eq!(text.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
    assert_eq!(value["message"], "Titre\nmessage\u{1b}[0m");
}
