use super::training_data::Batch;
use crate::tokenization::{ASSISTANT, EOS, Tokenizer, USER};
use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};

#[derive(PartialEq)]
pub struct Exchange {
    ids: Vec<u32>,
    assistant: usize,
}
#[derive(PartialEq)]
pub struct Dialogues {
    pub examples: Vec<Exchange>,
    pub tokens: usize,
    eos: u32,
}

impl Dialogues {
    pub fn parse(text: &str, tokenizer: &Tokenizer, sequence: usize, vocab: usize) -> Result<Self> {
        let user = tokenizer
            .token_to_id(USER)
            .ok_or_else(|| anyhow::anyhow!("Marqueur user absent"))?;
        let assistant = tokenizer
            .token_to_id(ASSISTANT)
            .ok_or_else(|| anyhow::anyhow!("Marqueur assistant absent"))?;
        let eos = tokenizer
            .token_to_id(EOS)
            .ok_or_else(|| anyhow::anyhow!("Marqueur end absent"))?;
        let mut examples = Vec::new();
        let mut tokens = 0;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let encoding = tokenizer
                .encode(line, false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let ids = encoding.get_ids().to_vec();
            ensure!(
                ids.len() > 4 && ids.len() <= sequence + 1,
                "Dialogue de {} tokens hors longueur 5..{} ; aucun tronquage silencieux",
                ids.len(),
                sequence + 1
            );
            ensure!(
                ids.first() == Some(&user) && ids.last() == Some(&eos),
                "Un echange complet user/assistant/end requis par ligne"
            );
            ensure!(
                ids.iter().all(|id| (*id as usize) < vocab),
                "Token hors vocabulaire"
            );
            for special in [user, assistant, eos] {
                ensure!(
                    ids.iter().filter(|&&id| id == special).count() == 1,
                    "Marqueur de role absent ou repete"
                );
            }
            let position = ids.iter().position(|id| *id == assistant).unwrap();
            ensure!(
                position > 1 && position + 2 < ids.len(),
                "Question et reponse non vides requises"
            );
            tokens += ids.len();
            examples.push(Exchange {
                ids,
                assistant: position,
            });
        }
        ensure!(!examples.is_empty(), "Corpus de dialogue vide");
        Ok(Self {
            examples,
            tokens,
            eos,
        })
    }

    pub fn batch(&self, indexes: &[usize], device: &Device) -> Result<Batch> {
        ensure!(!indexes.is_empty(), "Lot vide");
        let length = indexes
            .iter()
            .map(|&i| self.examples[i].ids.len() - 1)
            .max()
            .unwrap();
        let mut input = vec![self.eos; indexes.len() * length];
        let mut targets = Vec::new();
        let mut positions = Vec::new();
        for (row, &index) in indexes.iter().enumerate() {
            let example = &self.examples[index];
            let count = example.ids.len() - 1;
            input[row * length..row * length + count].copy_from_slice(&example.ids[..count]);
            // Le logit sur ASSISTANT predit le premier token de reponse ; EOS est supervise.
            for position in example.assistant..count {
                positions.push((row * length + position) as u32);
                targets.push(example.ids[position + 1]);
            }
        }
        let count = targets.len();
        let position_count = positions.len();
        Ok(Batch {
            input: Tensor::from_vec(input, (indexes.len(), length), device)?,
            target: Tensor::from_vec(targets, count, device)?,
            positions: Some(Tensor::from_vec(positions, position_count, device)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn targets_only_answers_and_end_not_prompts_or_padding() -> Result<()> {
        let data = Dialogues {
            eos: 2,
            tokens: 15,
            examples: vec![
                Exchange {
                    ids: vec![0, 10, 11, 1, 20, 21, 2],
                    assistant: 3,
                },
                Exchange {
                    ids: vec![0, 12, 1, 22, 2],
                    assistant: 2,
                },
            ],
        };
        let batch = data.batch(&[0, 1], &Device::Cpu)?;
        assert_eq!(batch.input.dims(), &[2, 6]);
        assert_eq!(batch.target.to_vec1::<u32>()?, [20, 21, 2, 22, 2]);
        assert_eq!(batch.positions.unwrap().to_vec1::<u32>()?, [3, 4, 5, 8, 9]);
        assert_eq!(batch.input.to_vec2::<u32>()?[1], [0, 12, 1, 22, 2, 2]);
        Ok(())
    }
}
