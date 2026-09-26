use serde::Serialize;
#[derive(Clone, Serialize)]
pub struct Scores {
    pub web: f32,
    pub foundation: f32,
}

pub fn accepts(initial: &Scores, current: &Scores, next: &Scores) -> bool {
    [
        initial.foundation,
        current.foundation,
        current.web,
        next.foundation,
        next.web,
    ]
    .into_iter()
    .all(f32::is_finite)
        && next.web + 0.0001 < current.web
        && next.foundation <= current.foundation * 1.02
        && next.foundation <= initial.foundation * 1.02
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prevent_cumulative_forgetting_and_non_finite_promotions() {
        let initial = Scores {
            web: 5.,
            foundation: 1.,
        };
        let current = Scores {
            web: 4.,
            foundation: 1.02,
        };
        assert!(accepts(
            &initial,
            &current,
            &Scores {
                web: 3.,
                foundation: 1.01
            }
        ));
        assert!(!accepts(
            &initial,
            &current,
            &Scores {
                web: 3.,
                foundation: 1.03
            }
        ));
        assert!(!accepts(
            &initial,
            &current,
            &Scores {
                web: f32::NAN,
                foundation: 1.
            }
        ));
        assert!(!accepts(
            &initial,
            &current,
            &Scores {
                web: 4.,
                foundation: 1.
            }
        ));
    }
}
