//! Reverses function executions and accumulates the results.
use std::collections::HashMap;

use crate::bad_trie::Builder;
use crate::bad_trie::Node;
use crate::Value;

#[derive(Debug, Clone, Copy)]
pub struct Input(u8);

pub struct SearchToken<'a> {
    state: &'a mut SearchState,
}

impl SearchToken<'_> {
    pub fn get(self, input: Input, index: u32) -> (Self, bool) {
        let ret = self.state.get(input.0, index);
        (self, ret)
    }

    pub fn eql(self, input: Input, value: impl AsRef<[u8]>) -> (Self, bool) {
        let bytes: &[u8] = value.as_ref();
        for (byte_idx, byte) in bytes.iter().copied().enumerate() {
            for bit_idx in 0..8 {
                let wanted = (byte >> bit_idx) & 1 != 0;
                if self.state.get(input.0, (8 * byte_idx + bit_idx) as u32) != wanted {
                    return (self, false);
                }
            }
        }

        (self, true)
    }
}

type Choice = (u8, u32, bool);

#[derive(Debug, Default)]
struct SearchState {
    cache: HashMap<(u8, u32), bool>,
    num_choices: usize,
    next_path_to_explore: Vec<Choice>,
    error: Option<&'static str>,
}

impl SearchState {
    fn new() -> SearchState {
        Default::default()
    }

    fn get(&mut self, input: u8, index: u32) -> bool {
        use std::collections::hash_map::Entry;

        let entry = self.cache.entry((input, index));
        if let Entry::Occupied(entry) = &entry {
            return *entry.get();
        }

        assert!(self.num_choices <= self.next_path_to_explore.len());
        if self.num_choices == self.next_path_to_explore.len() {
            self.next_path_to_explore.push((input, index, false));
        }

        let (expected_input, expected_index, value) = self.next_path_to_explore[self.num_choices];
        self.num_choices += 1;

        if (expected_input, expected_index) != (input, index) {
            self.error = Some("Nondeterministic search path");
        }

        *entry.or_insert(value)
    }

    fn advance_state(&mut self) {
        while let Some(back) = self.next_path_to_explore.last_mut() {
            if !back.2 {
                back.2 = true;
                break;
            }

            self.next_path_to_explore.pop();
        }

        self.cache.clear();
        self.num_choices = 0;
        self.error = None;
    }

    fn fathomed(&self) -> bool {
        self.next_path_to_explore.is_empty()
    }

    fn path(&self) -> &[(u8, u32, bool)] {
        &self.next_path_to_explore
    }
}

pub fn reverse_function<T: Value>(
    builder: &Builder<T>,
    count: u8,
    worker: impl for<'a> Fn(SearchToken<'a>, &[Input]) -> (SearchToken<'a>, T),
) -> Result<Node<T>, &'static str> {
    let inputs = (0..count).map(Input).collect::<Vec<Input>>();

    let mut acc = builder.make_empty();
    let mut state = SearchState::new();
    loop {
        let (_token, value) = worker(SearchToken { state: &mut state }, &inputs);
        let spine = builder.make_spine(state.path(), value);
        acc = builder.disjoint_union(acc, spine)?;

        state.advance_state();
        if state.fathomed() {
            break;
        }
    }

    Ok(acc)
}

pub fn map_reverse<T: Value + Send, Row: Send>(
    builder: &Builder<T>,
    items: impl rayon::iter::IntoParallelIterator<Item = Row>,
    param_count: u8,
    worker: impl for<'a> Fn(SearchToken<'a>, &[Input], &Row) -> (SearchToken<'a>, T) + Sync + Send,
) -> Result<Node<T>, &'static str> {
    use rayon::iter::ParallelIterator;

    let identity = || Ok(builder.make_empty());

    items
        .into_par_iter()
        .map(|row| {
            reverse_function(builder, param_count, |token, inputs| {
                worker(token, inputs, &row)
            })
        })
        .reduce(identity, |x, y| builder.merge(x?, y?))
}

#[cfg(test)]
mod test {
    use super::SearchToken;
    use crate::bad_trie::Builder;
    use crate::bad_trie::Node;
    use crate::Value;

    #[derive(Hash, PartialEq, Eq, Default, Clone, Debug)]
    struct Counter {
        count: usize,
    }

    impl merge::Merge for Counter {
        fn merge(&mut self, other: Counter) {
            self.count += other.count;
        }
    }

    impl Value for Counter {}

    #[test]
    fn test_reverse_smoke() {
        let builder = Builder::new();
        let cache =
            super::reverse_function(&builder, 2, |token, inputs| -> (SearchToken<'_>, Counter) {
                assert_eq!(inputs.len(), 2);
                let (token, x) = token.get(inputs[0], 1);
                let (token, y) = token.get(inputs[1], 0);

                (
                    token,
                    Counter {
                        count: (x as usize) + (y as usize),
                    },
                )
            })
            .expect("should work");

        println!("nodes: {:?}", &cache);
        assert_eq!(cache.lookup(&[&[0u8][..], &[0u8][..]]).unwrap(), None);
        assert_eq!(
            *cache.lookup(&[&[0u8][..], &[1u8][..]]).unwrap().unwrap(),
            Counter { count: 1 }
        );
        assert_eq!(cache.lookup(&[&[1u8][..], &[0u8][..]]).unwrap(), None);
        assert_eq!(
            *cache.lookup(&[&[1u8][..], &[1u8][..]]).unwrap().unwrap(),
            Counter { count: 1 }
        );
        assert_eq!(
            *cache.lookup(&[&[2u8][..], &[0u8][..]]).unwrap().unwrap(),
            Counter { count: 1 }
        );
        assert_eq!(
            *cache.lookup(&[&[2u8][..], &[1u8][..]]).unwrap().unwrap(),
            Counter { count: 2 }
        );
    }

    #[test]
    fn test_reverse_simplify_zero() {
        let builder = Builder::new();
        let cache =
            super::reverse_function(&builder, 2, |token, inputs| -> (SearchToken<'_>, Counter) {
                assert_eq!(inputs.len(), 2);
                let (token, _x) = token.get(inputs[0], 1);
                let (token, _y) = token.get(inputs[1], 0);

                (token, Counter { count: 0 })
            })
            .expect("should work");

        assert!(matches!(cache, Node::Default));
    }

    #[test]
    fn test_reverse_simplify_equal() {
        let builder = Builder::new();
        let cache =
            super::reverse_function(&builder, 2, |token, inputs| -> (SearchToken<'_>, Counter) {
                assert_eq!(inputs.len(), 2);
                let (token, _x) = token.get(inputs[0], 1);
                let (token, _y) = token.get(inputs[1], 0);

                (token, Counter { count: 1 })
            })
            .expect("should work");

        assert!(matches!(cache, Node::Leaf(_)));
        assert_eq!(
            *cache.lookup(&[&[0u8][..], &[0u8][..]]).unwrap().unwrap(),
            Counter { count: 1 }
        );
    }

    #[test]
    fn test_map_reverse() {
        let builder = Builder::new();
        let cache = super::map_reverse(
            &builder,
            vec![(1u8, 2usize), (2u8, 3usize), (1u8, 4usize)],
            1,
            |token, inputs, (key, value)| {
                let (token, matches) = token.eql(inputs[0], [*key]);
                let count = if matches { *value } else { 0 };
                (token, Counter { count })
            },
        )
        .unwrap();

        assert_eq!(cache.lookup(&[&[0u8]]).unwrap(), None);
        assert_eq!(
            *cache.lookup(&[&[1u8]]).unwrap().unwrap(),
            Counter { count: 6 }
        );
        assert_eq!(
            *cache.lookup(&[&[2u8]]).unwrap().unwrap(),
            Counter { count: 3 }
        );
        assert_eq!(cache.lookup(&[&[3u8]]).unwrap(), None);
    }
}
