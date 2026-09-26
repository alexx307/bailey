use super::{
    Parameter, StatefulAdamW,
    metadata::{Descriptor, Metadata},
    named_variables,
};
use anyhow::{Result, ensure};
use candle_core::{
    Tensor, Var,
    safetensors::{self, BufferedSafetensors},
};
use candle_nn::VarMap;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::Path,
};

impl StatefulAdamW {
    /// Cree exclusivement un nouveau dossier. Le manifeste est publie en dernier.
    pub fn save(&self, directory: &Path) -> Result<()> {
        self.hyperparameters.validate()?;
        let mut tensors = HashMap::new();
        for parameter in &self.parameters {
            for (prefix, moment) in [
                ("moment1", &parameter.first),
                ("moment2", &parameter.second),
            ] {
                check_finite(moment, &parameter.name)?;
                tensors.insert(
                    format!("{prefix}/{}", parameter.name),
                    moment.as_detached_tensor(),
                );
            }
        }
        fs::create_dir(directory)?;
        let moments_file = directory.join("moments.safetensors");
        safetensors::save(&tensors, &moments_file)?;
        let metadata = Metadata {
            version: 1,
            algorithm: "adamw".into(),
            step: self.step,
            hyperparameters: super::metadata::Hyperparameters {
                learning_rate: self.hyperparameters.learning_rate,
                ..super::metadata::Hyperparameters::new(self.hyperparameters.learning_rate)
            },
            parameters: self
                .parameters
                .iter()
                .map(|p| Descriptor::new(&p.name, &p.weight))
                .collect(),
            moments_sha256: digest(&fs::read(moments_file)?),
        };
        fs::write(
            directory.join("optimizer.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(())
    }

    /// Associe les moments aux variables deja chargees, sans modifier leurs poids.
    pub fn load(directory: &Path, vars: &VarMap) -> Result<Self> {
        let metadata: Metadata =
            serde_json::from_slice(&fs::read(directory.join("optimizer.json"))?)?;
        metadata.validate()?;
        let variables = named_variables(vars)?;
        let expected = variables
            .iter()
            .map(|(name, var)| Descriptor::new(name, var))
            .collect::<Vec<_>>();
        ensure!(
            metadata.parameters == expected,
            "Noms, formes ou types AdamW incompatibles"
        );
        let bytes = fs::read(directory.join("moments.safetensors"))?;
        ensure!(
            digest(&bytes) == metadata.moments_sha256,
            "Integrite des moments AdamW invalide"
        );
        let tensors = BufferedSafetensors::new(bytes)?;
        let actual_names = tensors
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<BTreeSet<_>>();
        let expected_names = variables
            .iter()
            .flat_map(|(name, _)| [format!("moment1/{name}"), format!("moment2/{name}")])
            .collect::<BTreeSet<_>>();
        ensure!(
            actual_names == expected_names,
            "Moments AdamW manquants ou supplementaires"
        );
        let mut parameters = Vec::with_capacity(variables.len());
        for (name, weight) in variables {
            let first = read_moment(&tensors, "moment1", &name, &weight, metadata.step)?;
            let second = read_moment(&tensors, "moment2", &name, &weight, metadata.step)?;
            ensure!(
                second
                    .min_all()?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    >= 0.,
                "Second moment AdamW negatif : {name}"
            );
            parameters.push(Parameter {
                name,
                weight,
                first,
                second,
            });
        }
        Ok(Self {
            parameters,
            step: metadata.step,
            hyperparameters: metadata.hyperparameters,
        })
    }
}

fn read_moment(
    tensors: &BufferedSafetensors,
    prefix: &str,
    name: &str,
    weight: &Var,
    step: usize,
) -> Result<Var> {
    let moment = tensors.load(&format!("{prefix}/{name}"), weight.device())?;
    ensure!(
        moment.shape() == weight.shape() && moment.dtype() == weight.dtype(),
        "Forme ou type du moment AdamW incompatible : {name}"
    );
    check_finite(&moment, name)?;
    if step == 0 {
        ensure!(
            moment
                .abs()?
                .max_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?
                == 0.,
            "Moments AdamW non nuls a l'etape zero : {name}"
        );
    }
    Ok(Var::from_tensor(&moment)?)
}

fn check_finite(tensor: &Tensor, name: &str) -> Result<()> {
    // La comparaison rejette a la fois NaN et les infinis, sans copier le tenseur au CPU.
    ensure!(
        tensor
            .abs()?
            .lt(f64::INFINITY)?
            .min_all()?
            .to_scalar::<u8>()?
            == 1,
        "Moment AdamW non fini : {name}"
    );
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
