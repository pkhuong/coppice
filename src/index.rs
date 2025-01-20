//! Index data structure based on running queries in reverse.
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use quinine::MonoBox;

use crate::bad_trie;
use crate::reverser::InverseContext;
use crate::reverser::SearchToken;
use crate::JoinKeys;
use crate::Value;

// Box and a worker lock.
type NodeCell<T> = Arc<(MonoBox<bad_trie::Node<T>>, Mutex<()>)>;

struct IndexedTable<Tablet: std::hash::Hash + Eq, Summary: Value> {
    builder: Arc<bad_trie::Builder<Summary>>,
    inverted_tablets: Mutex<HashMap<Tablet, NodeCell<Summary>>>,
}

impl<Tablet: std::hash::Hash + Eq, Summary: Value> IndexedTable<Tablet, Summary> {
    pub fn new() -> Result<Self, rayon::ThreadPoolBuildError> {
        Ok(IndexedTable {
            builder: Default::default(),
            inverted_tablets: Mutex::new(Default::default()),
        })
    }

    pub fn get_all_cells(&self, keys: Vec<Tablet>) -> Vec<NodeCell<Summary>> {
        let mut tablets = self.inverted_tablets.lock().unwrap();

        keys.into_iter()
            .map(|tablet| tablets.entry(tablet).or_default().clone())
            .collect::<Vec<_>>()
    }
}

fn ensure_populated_cells<Summary, Tablet, Params, Row, Rows, JoinKeys, RowFn, WorkerFn>(
    indexed_table: &IndexedTable<Tablet, Summary>,
    tablets: &[Tablet],
    params: &Params,
    join_keys: &JoinKeys,
    row_fn: &RowFn,
    worker: &WorkerFn,
) -> Result<Vec<NodeCell<Summary>>, &'static str>
where
    Summary: Value + Send,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send,
    Params: Sync,
    Row: Send,
    Rows: rayon::iter::IntoParallelIterator<Item = Row>,
    JoinKeys: Sync,
    RowFn: Fn(&Tablet) -> Result<Rows, &'static str> + Sync,
    WorkerFn: for<'b> Fn(
            SearchToken<'static, 'b>,
            &Params,
            &JoinKeys,
            &Row,
        ) -> (SearchToken<'static, 'b>, Summary)
        + Sync,
{
    let node_cells: Vec<NodeCell<Summary>> = indexed_table.get_all_cells(tablets.to_vec());
    let err: Mutex<Option<&'static str>> = Default::default();

    rayon::in_place_scope_fifo(|scope| {
        // Either spawn fresh work units to try and compute values,
        // or yield until they're all computed.
        loop {
            if err.lock().unwrap().is_some() {
                break;
            }

            let mut all_done = true;
            for (tablet, node_cell) in tablets.iter().zip(node_cells.iter()) {
                if node_cell.0.is_some() {
                    continue;
                }

                all_done = false;

                if node_cell.1.is_poisoned() {
                    node_cell.1.clear_poison();
                }

                // Someone's already working on it.
                if node_cell.1.try_lock().is_err() {
                    continue;
                }

                scope.spawn_fifo(|_scope| {
                    let Ok(_guard) = node_cell.1.try_lock() else {
                        return;
                    };

                    let rows = match row_fn(tablet) {
                        Ok(rows) => rows,
                        Err(e) => {
                            err.lock().unwrap().get_or_insert(e);
                            return;
                        }
                    };

                    let result = crate::reverser::map_reverse(
                        &indexed_table.builder,
                        rows,
                        &join_keys,
                        |token, join_keys, row| worker(token, params, join_keys, row),
                    );

                    match result {
                        Ok(result) => {
                            let _ = node_cell.0.store(Box::new(result));
                        }
                        Err(e) => {
                            let _ = err.lock().unwrap().get_or_insert(e);
                        }
                    }
                });
            }

            if all_done {
                assert!(node_cells
                    .iter()
                    .all(|node_cell| node_cell.0.as_ref().is_some()));
                break;
            }

            // Pull some work if we can.
            for _ in tablets.iter() {
                if rayon::yield_local().is_none() {
                    break;
                }
            }

            if rayon::yield_now().is_none() {
                std::thread::yield_now(); // better than nothing
            }
        }
    });

    let err = err.lock().unwrap();
    if let Some(e) = *err {
        Err(e)
    } else {
        assert!(node_cells
            .iter()
            .all(|node_cell| node_cell.0.as_ref().is_some()));
        Ok(node_cells)
    }
}

pub fn aggregate<Summary, Tablet, Params, Row, Rows, JoinKeysT, RowFn, WorkerFn>(
    tablets: &[Tablet],
    params: Params,
    join_keys: JoinKeysT,
    row_fn: RowFn,
    worker: WorkerFn,
) -> Result<Summary, &'static str>
where
    Summary: Value + Send,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send,
    Params: Sync,
    Row: Send,
    Rows: rayon::iter::IntoParallelIterator<Item = Row>,
    JoinKeysT: JoinKeys,
    RowFn: Fn(&Tablet) -> Result<Rows, &'static str> + Sync,
    WorkerFn: for<'a, 'b> Fn(
            SearchToken<'a, 'b>,
            &Params,
            &JoinKeysT::Ret<'a>,
            &Row,
        ) -> (SearchToken<'a, 'b>, Summary)
        + Sync,
{
    let mut ctx = InverseContext::<'static>::new();
    let join_keys = join_keys.invert(&mut ctx)?;

    let index: IndexedTable<Tablet, Summary> = IndexedTable::new().unwrap();

    let node_cells =
        ensure_populated_cells(&index, tablets, &params, &join_keys, &row_fn, &worker)?;
    let lookup_keys = ctx.keys();
    let mut acc: Option<Summary> = None;
    for node_cell in node_cells {
        if let Some(value) = node_cell
            .0
            .as_ref()
            .expect("already forced")
            .lookup(&lookup_keys)?
        {
            let value = (*value).clone();
            match &mut acc {
                Some(ref mut acc) => acc.merge(value),
                None => acc = Some(value),
            }
        }
    }

    Ok(acc.unwrap_or_default())
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
            &["foo"],
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

        assert_eq!(value, Counter { count: 6 });
    }
}
