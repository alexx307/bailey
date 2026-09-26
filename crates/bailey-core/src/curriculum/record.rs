use crate::tokenization::{ASSISTANT, EOS, USER};

pub(super) struct Unit {
    pub id: String,
    pub topic: &'static str,
    pub text: String,
}

pub(super) fn prose(topic: &'static str, samples: &[&str], units: &mut Vec<Unit>) {
    for (index, text) in samples.iter().enumerate() {
        units.push(Unit {
            id: format!("{topic}-{:03}", index + 1),
            topic,
            text: format!("{text}{EOS}\n"),
        });
    }
}

pub(super) fn dialogue(samples: &[(&str, &str)], units: &mut Vec<Unit>) {
    for (index, (prompt, answer)) in samples.iter().enumerate() {
        units.push(Unit {
            id: format!("dialogue-{:03}", index + 1),
            topic: "dialogue",
            text: format!("{USER}{prompt}{ASSISTANT}{answer}{EOS}\n"),
        });
    }
}
