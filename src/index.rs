//! Index data structure based on running queries in reverse.
use std::collections::HashMap;
use std::sync::Arc;

use crate::bad_trie::Node;
use crate::reverser::InverseContext;
use crate::reverser::SearchToken;
use crate::JoinKeys;
use crate::Value;

struct Index<Tablet: std::hash::Hash + Eq, Summary: Value> {
    inverted_tablets: HashMap<Tablet, Node<Summary>>,
}

pub fn aggregate<Summary, Tablet, Params, Row, Rows, JoinKeysT, RowFn, WorkerFn>(
    tablet: Tablet,
    params: Params,
    join_keys: JoinKeysT,
    row_fn: RowFn,
    worker: WorkerFn,
) -> Result<Option<Arc<Summary>>, &'static str>
where
    Summary: Value + Send,
    Tablet: std::hash::Hash + Eq + Clone,
    Params: Sync,
    Row: Send,
    Rows: rayon::iter::IntoParallelIterator<Item = Row>,
    JoinKeysT: JoinKeys,
    RowFn: FnOnce(&Tablet) -> Result<Rows, &'static str>,
    WorkerFn: for<'a, 'b> Fn(
            SearchToken<'a, 'b>,
            &Params,
            &JoinKeysT::Ret<'a>,
            &Row,
        ) -> (SearchToken<'a, 'b>, Summary)
        + Sync
        + Send,
{
    use std::collections::hash_map::Entry;

    let mut ctx = InverseContext::<'static>::new();
    let join_keys = join_keys.invert(&mut ctx);

    let mut caches: Index<Tablet, Summary> = Index {
        inverted_tablets: Default::default(),
    };

    let entry = caches.inverted_tablets.entry(tablet.clone());
    let cache = {
        match entry {
            Entry::Occupied(ref entry) => entry.get(),
            Entry::Vacant(entry) => {
                let builder = crate::bad_trie::Builder::new();
                let result = crate::reverser::map_reverse(
                    &builder,
                    row_fn(&tablet)?,
                    &join_keys,
                    |token, join_keys, row| worker(token, &params, join_keys, row),
                )?;
                entry.insert(result)
            }
        }
    };

    cache.lookup(&ctx.keys())
}

#[cfg(test)]
mod test {
    use super::*;

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
    fn test_agg() {
        let value = aggregate(
            "foo",
            ("test",),
            1u8,
            |_foo| Ok(vec![(1u8, 2usize), (2u8, 3usize), (1u8, 4usize)]),
            |token, params, needle, (key, value)| {
                assert_eq!(*params, ("test",));
                let (token, matches) = token.eql(needle, key);
                let count = if matches { *value } else { 0 };
                (token, Counter { count })
            },
        )
        .unwrap();

        assert_eq!(*value.unwrap(), Counter { count: 6 });
    }
}
