use super::{
    cli::{Cli, Command, DeviceChoice},
    probe,
};
use crate::{
    model::CoreConfig,
    training::{self, TrainConfig},
};
use anyhow::{Result, ensure};
use candle_core::Device;
use std::{fs, path::Path};

pub fn read_model(file: &Path) -> Result<CoreConfig> {
    let config: CoreConfig = serde_json::from_slice(&fs::read(file)?)?;
    config.validate()?;
    Ok(config)
}

fn device(choice: DeviceChoice) -> Result<Device> {
    match choice {
        DeviceChoice::Cpu => Ok(Device::Cpu),
        DeviceChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::new_cuda(0)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("Compiler avec --features cuda pour utiliser la RTX")
            }
        }
    }
}

pub fn execute(cli: Cli) -> Result<()> {
    match cli.command {
        Command::TokenizerMix { config, out } => crate::tokenization::mix::prepare(&config, &out),
        Command::TokenizerAudit {
            tokenizer,
            baseline,
            probes,
            out,
        } => crate::tokenization::audit::audit(&tokenizer, baseline.as_deref(), &probes, &out),
        Command::HfImport(args) => crate::forge::huggingface::import(&args),
        Command::CurateText { data, out } => crate::forge::curation::curate(&data, &out),
        Command::ResearchTopic { topic } => {
            for article in
                crate::research::search_and_store(&topic, "fr", Path::new("data/library"), 3)?
            {
                println!("{:?}\n{}", article.title, article.url);
            }
            Ok(())
        }
        Command::Research(args) => {
            let config = serde_json::from_slice(&fs::read(&args.config)?)?;
            let mut learner = crate::learning::Learner::new(
                args.init_from,
                &args.rehearsal,
                &args.out,
                args.steps,
                &config,
            )?;
            let device = device(cli.device)?;
            crate::research::run_with_hook(&config, &args.library, &args.out, |data, cycle| {
                learner.cycle(data, cycle, &device)
            })
        }
        Command::Info { config } => {
            let model = read_model(&config)?;
            let count = model.parameter_count()?;
            println!(
                "Bailey Core | {count} parametres | {} couches | contexte {} tokens",
                model.num_layers, model.max_seq_len
            );
            println!(
                "GQA {} Q / {} KV | RoPE | RMSNorm | SwiGLU | embeddings partages",
                model.num_attention_heads, model.num_kv_heads
            );
            println!(
                "Poids FP32 {:.1} Mio ; poids + gradients + Adam {:.2} Gio AVANT activations/temporaires.",
                count as f64 * 4.0 / 1048576.0,
                count as f64 * 16.0 / 1073741824.0
            );
            println!(
                "Architecture definie ; aucune competence de langage supposee avant entrainement."
            );
            Ok(())
        }
        Command::PrepareSeed { out } => crate::curriculum::prepare(&out),
        Command::TokenizerTrain {
            train,
            out,
            vocab_size,
        } => crate::tokenization::train(&train, &out, vocab_size),
        Command::CheckModel {
            config,
            sequence,
            steps,
            report,
        } => probe::run(
            &read_model(&config)?,
            sequence,
            steps,
            &report,
            &device(cli.device)?,
        ),
        Command::Train(args) => {
            ensure!(
                !args.tiny || args.init_from.is_none(),
                "--tiny et --init-from ne se combinent pas"
            );
            let model = if args.tiny {
                CoreConfig::tiny(crate::tokenization::load(&args.tokenizer)?.get_vocab_size(true))
            } else {
                read_model(&args.config)?
            };
            training::train_until(
                TrainConfig {
                    model,
                    data: args.data,
                    tokenizer: args.tokenizer,
                    steps: args.steps,
                    sequence: args.sequence,
                    batch_size: args.batch_size,
                    learning_rate: args.learning_rate,
                    eval_every: args.eval_every,
                    seed: args.seed,
                    init_from: args.init_from,
                    warmup_steps: args.warmup_steps,
                    min_lr_ratio: args.min_lr_ratio,
                    max_grad_norm: Some(args.max_grad_norm),
                    evaluation_windows: args.evaluation_windows,
                    objective: args.objective,
                    data_format: args.data_format,
                },
                &args.out,
                &device(cli.device)?,
                args.stop_after,
            )
            .map(|_| ())
        }
        Command::Resume {
            run,
            out,
            data,
            stop_after,
        } => training::resume(
            &run,
            &out,
            data.as_deref(),
            &device(cli.device)?,
            stop_after,
        )
        .map(|_| ()),
        Command::ForgeBuild {
            data,
            tokenizer,
            out,
            shard_tokens,
        } => {
            let manifest = crate::forge::dataset::build(&data, &tokenizer, &out, shard_tokens)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::ForgeInfo { data } => {
            let manifest = crate::forge::dataset::read_manifest(&data)?;
            println!(
                "Tokenizer : {} tokens de vocabulaire",
                manifest.tokenizer.vocab_size
            );
            for p in manifest.partitions {
                println!(
                    "{} : {} tokens de corpus, {} shards, {} octets U32",
                    p.name,
                    p.tokens,
                    p.shards.len(),
                    p.tokens * 4
                );
            }
            Ok(())
        }
        Command::TokenizerInfo { tokenizer } => {
            let tokens = crate::tokenization::load(&tokenizer)?;
            println!(
                "Vocabulaire reel : {} tokens. Il s'agit du dictionnaire, pas du nombre de tokens du corpus.",
                tokens.get_vocab_size(true)
            );
            Ok(())
        }
        Command::Generate {
            run,
            prompt,
            tokens,
            sampling,
            diagnostics,
        } => {
            if let Some(path) = &diagnostics {
                ensure!(!path.exists(), "Rapport deja existant");
            }
            let device = device(cli.device)?;
            let (model, config) = training::checkpoint::load(&run, &device)?;
            let tokenizer = crate::tokenization::load(&run.join("tokenizer.json"))?;
            let result = crate::inference::generate_with(
                &model,
                &config.model,
                &tokenizer,
                &prompt,
                tokens,
                &sampling,
                &device,
            )?;
            println!("{}", result.text);
            if let Some(path) = diagnostics {
                fs::write(path, serde_json::to_vec_pretty(&result)?)?;
            }
            Ok(())
        }
        Command::Evaluate { run, file } => {
            let device = device(cli.device)?;
            let (model, config) = training::checkpoint::load(&run, &device)?;
            let tokenizer = crate::tokenization::load(&run.join("tokenizer.json"))?;
            let loss =
                training::evaluation::evaluate_file(&model, &file, &tokenizer, &config, &device)?;
            println!(
                "Perte de prediction, objectif {} sur {} elements fixes au maximum : {loss:.4}. Ce n'est pas un score de conversation.",
                serde_json::to_string(&config.objective)?,
                config.evaluation_windows
            );
            Ok(())
        }
        Command::Console { run } => crate::inference::console(&run, &device(cli.device)?),
        Command::PrepareDialogue { source, out } => {
            crate::curriculum::dialogue_course::prepare(&source, &out)
        }
        Command::DialogueReport {
            run,
            prompts,
            out,
            tokens,
        } => super::dialogue_report::run(&run, &prompts, &out, tokens, &device(cli.device)?),
    }
}
