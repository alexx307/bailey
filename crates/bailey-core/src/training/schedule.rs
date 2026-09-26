use super::TrainConfig;

pub fn learning_rate(config: &TrainConfig, step: usize) -> f64 {
    if config.warmup_steps > 0 && step <= config.warmup_steps {
        return config.learning_rate * step as f64 / config.warmup_steps as f64;
    }
    let span = config.steps.saturating_sub(config.warmup_steps).max(1);
    let progress = step.saturating_sub(config.warmup_steps).min(span) as f64 / span as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    config.learning_rate * (config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CoreConfig;
    use serde_json::json;

    #[test]
    fn old_configs_keep_constant_rate_and_warmup_reaches_the_peak() -> anyhow::Result<()> {
        let mut config: TrainConfig = serde_json::from_value(json!({
            "model":CoreConfig::tiny(300), "data":"data", "tokenizer":"tokens",
            "steps":100,"sequence":16,"batch_size":1,"learning_rate":0.001,
            "eval_every":20,"seed":42,"init_from":null
        }))?;
        config.validate()?;
        assert_eq!(config.evaluation_windows, 4);
        assert_eq!(config.max_grad_norm, None);
        assert_eq!(learning_rate(&config, 1), 0.001);
        assert_eq!(learning_rate(&config, 100), 0.001);
        config.warmup_steps = 10;
        config.min_lr_ratio = 0.1;
        assert!((learning_rate(&config, 1) - 0.0001).abs() < 1e-12);
        assert!((learning_rate(&config, 10) - 0.001).abs() < 1e-12);
        assert!((learning_rate(&config, 100) - 0.0001).abs() < 1e-12);
        for step in 11..100 {
            assert!(learning_rate(&config, step + 1) <= learning_rate(&config, step));
        }
        config.max_grad_norm = Some(f64::NAN);
        assert!(config.validate().is_err());
        Ok(())
    }
}
