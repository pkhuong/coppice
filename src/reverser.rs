//! Reverses function executions and accumulates the results.
use std::collections::HashMap;

use crate::bad_trie::Builder;
use crate::bad_trie::Node;
use crate::Aggregate;
use crate::BaseJoinKey;

#[derive(Debug)]
pub struct Inverse<'tag, T: BaseJoinKey + ?Sized>(
    u8,
    std::marker::PhantomData<fn(&'tag T) -> &'tag T>,
); // Invariant over 'tag

#[derive(Debug, Default)]
pub struct InverseContext<'tag> {
    values: Vec<Box<[u8]>>,
    _invariant: std::marker::PhantomData<fn(&'tag ()) -> &'tag ()>, // Invariant over 'tag
}

impl<'tag> InverseContext<'tag> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn fake<T: BaseJoinKey + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<Inverse<'tag, T>, &'static str> {
        if self.values.len() > u8::MAX as usize {
            return Err("too many join keys for context");
        }

        let idx = self.values.len() as u8;
        self.values.push(Box::from(value.to_bytes().as_ref()));
        Ok(Inverse(idx, Default::default()))
    }

    pub fn keys(&self) -> Vec<&[u8]> {
        self.values.iter().map(|x| &**x).collect::<Vec<_>>()
    }
}

pub struct SearchToken<'tag, 'a> {
    state: &'a mut SearchState,
    // Make the lifetime invariant
    _marker: std::marker::PhantomData<fn(&'tag ()) -> &'tag ()>,
}

impl<'tag> SearchToken<'tag, '_> {
    pub fn get<T: BaseJoinKey + ?Sized>(
        self,
        input: &Inverse<'tag, T>,
        index: u32,
    ) -> (Self, bool) {
        self.state
            .check_mapping(input.0, input as *const Inverse<_> as usize);
        let ret = self.state.get(input.0, index);
        (self, ret)
    }

    fn eql_raw<T: BaseJoinKey + ?Sized>(
        self,
        input: &Inverse<'tag, T>,
        bytes: &[u8],
    ) -> (Self, bool) {
        self.state
            .check_mapping(input.0, input as *const Inverse<_> as usize);
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

    pub fn eql<T: BaseJoinKey + ?Sized>(self, input: &Inverse<'tag, T>, value: &T) -> (Self, bool) {
        let value = value.to_bytes();
        self.eql_raw(input, value.as_ref())
    }

    pub fn eql_any<T: BaseJoinKey + ?Sized>(
        self,
        input: &Inverse<'tag, T>,
        values: &[&T],
    ) -> (Self, bool) {
        fn get_bit(bytes: &[u8], idx: usize) -> bool {
            let byte_index = idx / 8;
            let sub_idx = idx % 8;

            (bytes.get(byte_index).unwrap_or(&0) & (1 << sub_idx)) != 0
        }

        self.state
            .check_mapping(input.0, input as *const Inverse<_> as usize);
        let values = values.iter().map(|x| x.to_bytes()).collect::<Vec<_>>();
        let mut raw_values: Vec<&[u8]> = values.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
        raw_values.sort();
        raw_values.dedup();

        let mut next_bit_idx: u32 = 0;
        while raw_values.len() > 1 {
            let actual_bit = self.state.get(input.0, next_bit_idx);
            raw_values.retain(|bytes| get_bit(bytes, next_bit_idx as usize) == actual_bit);

            next_bit_idx += 1;
        }

        if let Some(highlander) = raw_values.first() {
            self.eql_raw(input, highlander)
        } else {
            (self, false)
        }
    }
}

type Choice = (u8, u32, bool);

#[derive(Debug, Default)]
struct SearchState {
    index_mapping: HashMap<u8, usize>,
    cache: HashMap<(u8, u32), bool>,
    num_choices: usize,
    next_path_to_explore: Vec<Choice>,
    error: Option<&'static str>,
}

impl SearchState {
    fn new() -> SearchState {
        Default::default()
    }

    fn check_mapping(&mut self, input: u8, addr: usize) {
        if *self.index_mapping.entry(input).or_insert(addr) != addr {
            self.error = Some("invalid mapping for input join key");
        }
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

pub fn reverse_function<T: Aggregate, JK>(
    builder: &Builder<T>,
    join_keys: &JK,
    worker: impl for<'b> Fn(SearchToken<'static, 'b>, &JK) -> T,
) -> Result<Node<T>, &'static str> {
    let mut acc = builder.make_neutral();
    let mut state = SearchState::new();
    loop {
        let token = SearchToken {
            state: &mut state,
            _marker: Default::default(),
        };
        let value = worker(token, join_keys);
        let spine = builder.make_spine(state.path(), value);
        acc = builder.disjoint_union(acc, spine)?;

        if let Some(e) = state.error {
            return Err(e);
        }

        state.advance_state();
        if state.fathomed() {
            break;
        }
    }

    Ok(acc)
}

pub fn map_reverse<T: Aggregate + Send, Row: Send, JK: Sync>(
    builder: &Builder<T>,
    items: impl rayon::iter::IntoParallelIterator<Item = Row>,
    join_keys: &JK,
    worker: impl for<'b> Fn(SearchToken<'static, 'b>, &JK, &Row) -> T + Sync + Send,
) -> Result<Node<T>, &'static str> {
    use rayon::iter::ParallelIterator;

    let identity = || builder.make_neutral();

    items
        .into_par_iter()
        .map(|row| {
            reverse_function(builder, join_keys, |token, inputs| {
                worker(token, inputs, &row)
            })
        })
        .try_reduce(identity, |x, y| builder.merge(x, y))
}

#[cfg(test)]
mod test {
    use super::InverseContext;
    use crate::aggregates::Counter;
    use crate::bad_trie::Builder;
    use crate::bad_trie::Node;

    #[test]
    fn test_reverse_smoke() {
        let builder: Builder<_> = Default::default();
        let mut ctx = InverseContext::new();
        let cache = super::reverse_function(
            &builder,
            &[ctx.fake(&0u8).unwrap(), ctx.fake(&0u8).unwrap()],
            |token, inputs| -> Counter {
                assert_eq!(inputs.len(), 2);
                let (token, x) = token.get(&inputs[0], 1);
                let (_token, y) = token.get(&inputs[1], 0);

                Counter::new((x as u64) + (y as u64))
            },
        )
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
        let builder: Builder<_> = Default::default();
        let mut ctx = InverseContext::new();
        let cache = super::reverse_function(
            &builder,
            &[ctx.fake(&0u8).unwrap(), ctx.fake(&0u8).unwrap()],
            |token, inputs| -> Counter {
                assert_eq!(inputs.len(), 2);
                let (token, _x) = token.get(&inputs[0], 1);
                let (_token, _y) = token.get(&inputs[1], 0);

                Counter { count: 0 }
            },
        )
        .expect("should work");

        assert!(matches!(cache, Node::Neutral));
    }

    #[test]
    fn test_reverse_simplify_equal() {
        let builder: Builder<_> = Default::default();
        let mut ctx = InverseContext::new();
        let cache = super::reverse_function(
            &builder,
            &[ctx.fake(&0u8).unwrap(), ctx.fake(&0u8).unwrap()],
            |token, inputs| -> Counter {
                assert_eq!(inputs.len(), 2);
                let (token, _x) = token.get(&inputs[0], 1);
                let (_token, _y) = token.get(&inputs[1], 0);

                Counter { count: 1 }
            },
        )
        .expect("should work");

        assert!(matches!(cache, Node::Leaf(_)));
        assert_eq!(
            *cache.lookup(&[&[0u8][..], &[0u8][..]]).unwrap().unwrap(),
            Counter { count: 1 }
        );
    }

    #[test]
    fn test_map_reverse() {
        let builder: Builder<_> = Default::default();
        let mut ctx = InverseContext::new();
        let cache = super::map_reverse(
            &builder,
            vec![(1u8, 2usize), (2u8, 3usize), (1u8, 4usize)],
            &ctx.fake(&0u8).unwrap(),
            |token, needle, (key, value)| {
                let (_token, matches) = token.eql(needle, key);
                let count = if matches { *value } else { 0 };
                Counter::new(count as u64)
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
