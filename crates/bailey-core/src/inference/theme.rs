//! Habillage texte de la console : art ASCII, couleurs ANSI, panneaux encadres.
//! Aucune donnee affichee ici n'est inventee ; les valeurs viennent de l'appelant.

const CYAN: &str = "\x1b[1;36m";
const GREEN: &str = "\x1b[1;32m";
const DIM: &str = "\x1b[90m";
const RESET: &str = "\x1b[0m";

const TITLE: [&str; 6] = [
    "██████╗  █████╗ ██╗██╗     ███████╗██╗   ██╗",
    "██╔══██╗██╔══██╗██║██║     ██╔════╝╚██╗ ██╔╝",
    "██████╔╝███████║██║██║     █████╗   ╚████╔╝ ",
    "██╔══██╗██╔══██║██║██║     ██╔══╝    ╚██╔╝  ",
    "██████╔╝██║  ██║██║███████╗███████╗   ██║   ",
    "╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝   ╚═╝   ",
];

pub fn title() {
    for line in TITLE {
        println!("{CYAN}{line}{RESET}");
    }
    println!("{DIM}CLI — modele de langue experimental, local.{RESET}");
    println!();
}

/// Encadre `rows` (deja formatees "cle : valeur") sous un titre, largeur ajustee
/// au contenu. N'affiche que ce qui est passe ; ne complete jamais une valeur.
pub fn panel(title: &str, rows: &[(&str, String)]) {
    let label_width = rows
        .iter()
        .map(|(k, _)| k.chars().count())
        .max()
        .unwrap_or(0);
    let lines: Vec<String> = rows
        .iter()
        .map(|(key, value)| format!("{key:<label_width$} : {value}"))
        .collect();
    let inner = lines
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0)
        .max(title.chars().count() + 2);
    println!(
        "{DIM}┌─ {CYAN}{title}{DIM} {}┐{RESET}",
        "─".repeat(inner.saturating_sub(title.chars().count() + 1))
    );
    for line in &lines {
        println!("{DIM}│{RESET} {line:<inner$} {DIM}│{RESET}");
    }
    println!("{DIM}└{}┘{RESET}", "─".repeat(inner + 2));
}

pub fn you_prompt() -> String {
    format!("{GREEN}Toi >{RESET} ")
}

pub fn bailey_prefix() -> String {
    format!("{CYAN}Bailey >{RESET}")
}
