use crate::{
    tokenization::{self, ASSISTANT, USER},
    training::checkpoint,
};
use anyhow::Result;
use candle_core::Device;
use std::{
    io::{self, Write},
    path::Path,
};

pub fn console(run: &Path, device: &Device) -> Result<()> {
    let (model, config) = checkpoint::load(run, device)?;
    let tokenizer = tokenization::load(&run.join("tokenizer.json"))?;
    println!(
        "Bailey Core | {} parametres | {}",
        config.model.parameter_count()?,
        run.display()
    );
    println!("Modele experimental : un essai court ne lui apprend pas encore a converser.");
    println!("/brut : completion ; /dialogue : format question/reponse ; /quitter : sortie");
    println!("/recherche sujet : Wikipedia ; /memoire sujet : sources deja conservees");
    let mut dialogue = true;
    let mut sampling = super::Sampling::default();
    let mut diagnostic = false;
    println!("Calcul sur {device:?} | /greedy ; /sampling ; /diagnostic");
    println!("Cette console ne s'entraine pas en attendant tes messages.");
    loop {
        print!("Toi > ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let text = line.trim();
        if let Some(query) = text.strip_prefix("/recherche ") {
            match crate::research::search_and_store(query, "fr", Path::new("data/library"), 3) {
                Ok(articles) => show_sources(&articles),
                Err(error) => eprintln!("Recherche : {error:#}"),
            }
            continue;
        }
        if let Some(query) = text.strip_prefix("/memoire ") {
            match crate::research::lookup(query, Path::new("data/library"), 3) {
                Ok(articles) => show_sources(&articles),
                Err(error) => eprintln!("Memoire : {error:#}"),
            }
            continue;
        }
        match text {
            "" => continue,
            "/quitter" => break,
            "/greedy" => {
                sampling = super::Sampling::default();
                println!("Generation deterministe, sans penalite de repetition.");
                continue;
            }
            "/sampling" => {
                sampling = super::Sampling {
                    temperature: 0.7,
                    repetition_penalty: 1.1,
                    ..Default::default()
                };
                println!(
                    "Temperature 0.7, top-k 40, top-p 0.9, repetition 1.1. Ceci n'ajoute aucune connaissance."
                );
                continue;
            }
            "/diagnostic" => {
                diagnostic = !diagnostic;
                println!("Diagnostic des premieres predictions : {diagnostic}");
                continue;
            }
            "/brut" => {
                dialogue = false;
                println!("Completion libre.");
                continue;
            }
            "/dialogue" => {
                dialogue = true;
                println!("Question/reponse sans historique.");
                continue;
            }
            _ => {}
        }
        let prompt = if dialogue {
            format!("{USER}{text}{ASSISTANT}")
        } else {
            text.to_owned()
        };
        match super::generate_with(
            &model,
            &config.model,
            &tokenizer,
            &prompt,
            96,
            &sampling,
            device,
        ) {
            Ok(result) => {
                println!("Bailey > {}\n", result.text);
                if diagnostic {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
            Err(error) => eprintln!("Erreur : {error:#}"),
        }
    }
    Ok(())
}

fn show_sources(articles: &[crate::web::Article]) {
    for article in articles {
        let excerpt: String = article
            .text
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .take(420)
            .collect();
        println!("Source {:?}\n{}\n{}\n", article.title, article.url, excerpt);
    }
    println!("Lecture documentaire : les poids ne sont pas modifies par cette commande.");
}
