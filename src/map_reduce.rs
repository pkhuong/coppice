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

struct SummaryCache<Tablet: std::hash::Hash + Eq, Summary: Value> {
    builder: Arc<bad_trie::Builder<Summary>>,
    inverted_summaries: Mutex<HashMap<Tablet, NodeCell<Summary>>>,
}

impl<Tablet: std::hash::Hash + Eq, Summary: Value> SummaryCache<Tablet, Summary> {
    pub fn new() -> Self {
        SummaryCache {
            builder: Default::default(),
            inverted_summaries: Mutex::new(Default::default()),
        }
    }

    pub fn get_all_cells(&self, keys: Vec<Tablet>) -> Vec<NodeCell<Summary>> {
        let mut tablets = self.inverted_summaries.lock().unwrap();

        keys.into_iter()
            .map(|tablet| tablets.entry(tablet).or_default().clone())
            .collect::<Vec<_>>()
    }
}

// Summary, Tablet, Params, JoinKeys, RowFn, WorkerFn
type ShapeKey = [std::any::TypeId; 6];

static SUMMARY_CACHES: std::sync::LazyLock<
    Mutex<HashMap<ShapeKey, Arc<dyn std::any::Any + Sync + Send>>>,
> = std::sync::LazyLock::new(Default::default);

pub fn clear_all_summary_caches() {
    SUMMARY_CACHES.lock().unwrap().clear();
}

fn ensure_populated_cells<Summary, Tablet, Params, Row, Rows, JoinKeys, RowFn, WorkerFn>(
    tablets: &[Tablet],
    params: &Params,
    join_keys: &JoinKeys,
    row_fn: &RowFn,
    worker: &WorkerFn,
) -> Result<Vec<NodeCell<Summary>>, &'static str>
where
    Summary: Value + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: Sync + 'static,
    Row: Send,
    Rows: rayon::iter::IntoParallelIterator<Item = Row>,
    JoinKeys: Sync + 'static,
    RowFn: Fn(&Tablet) -> Result<Rows, &'static str> + Sync + 'static,
    WorkerFn: for<'b> Fn(
            SearchToken<'static, 'b>,
            &Params,
            &JoinKeys,
            &Row,
        ) -> (SearchToken<'static, 'b>, Summary)
        + Sync
        + 'static,
{
    use std::any::TypeId;

    let shape_key: ShapeKey = [
        TypeId::of::<Summary>(),
        TypeId::of::<Tablet>(),
        TypeId::of::<Params>(),
        TypeId::of::<JoinKeys>(),
        TypeId::of::<RowFn>(),
        TypeId::of::<WorkerFn>(),
    ];
    let summary_cache = SUMMARY_CACHES
        .lock()
        .unwrap()
        .entry(shape_key)
        .or_insert_with(|| Arc::new(SummaryCache::<Tablet, Summary>::new()))
        .clone();

    let summary_cache = summary_cache
        .downcast_ref::<SummaryCache<Tablet, Summary>>()
        .expect("types are part of the shape key");

    let node_cells: Vec<NodeCell<Summary>> = summary_cache.get_all_cells(tablets.to_vec());
    let builder = &summary_cache.builder;
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
                        builder,
                        rows,
                        join_keys,
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

pub fn map_reduce<Summary, Tablet, Params, Row, Rows, JoinKeysT, RowFn, WorkerFn>(
    tablets: &[Tablet],
    params: Params,
    join_keys: JoinKeysT,
    row_fn: RowFn,
    worker: WorkerFn,
) -> Result<Summary, &'static str>
where
    Summary: Value + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: Sync + 'static,
    Row: Send,
    Rows: rayon::iter::IntoParallelIterator<Item = Row>,
    JoinKeysT: JoinKeys + 'static,
    RowFn: Fn(&Tablet) -> Result<Rows, &'static str> + Sync + 'static,
    WorkerFn: for<'a, 'b> Fn(
            SearchToken<'a, 'b>,
            &Params,
            &JoinKeysT::Ret<'a>,
            &Row,
        ) -> (SearchToken<'a, 'b>, Summary)
        + Sync
        + 'static,
{
    let mut ctx = InverseContext::<'static>::new();
    let join_keys = join_keys.invert(&mut ctx)?;

    let node_cells = ensure_populated_cells(tablets, &params, &join_keys, &row_fn, &worker)?;
    let lookup_keys = ctx.keys();
    let mut acc: Option<Summary> = None;
    for node_cell in node_cells {
        if let Some(value) = node_cell
            .0
            .as_ref()
            .expect("already forced by ensure_populated_cells")
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

    static LOAD_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn dummy(tablets: &[String], join_key: u8) -> Result<Counter, &'static str> {
        map_reduce(
            tablets,
            ("test",),
            join_key,
            |tablet| {
                LOAD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                if tablet == "foo" {
                    Ok(vec![(1u8, 2usize), (2u8, 3usize), (1u8, 4usize)])
                } else {
                    Ok(vec![
                        (1u8, 20usize),
                        (2u8, 30usize),
                        (1u8, 40usize),
                        (11u8, 42usize),
                    ])
                }
            },
            |token, params, needle, (key, value)| {
                assert_eq!(*params, ("test",));
                let (token, matches) = token.eql(needle, key);
                let count = if matches { *value } else { 0 };
                (token, Counter { count })
            },
        )
    }

    use rusty_fork::rusty_fork_test;
    rusty_fork_test! {
        #[test]
        fn test_agg_smoke() {
            LOAD_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);

            assert_eq!(dummy(&["foo".to_owned()], 1).unwrap(), Counter { count: 6 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 1);

            assert_eq!(dummy(&["bar".to_owned()], 1).unwrap(), Counter { count: 60 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 2);

            assert_eq!(dummy(&["foo".to_owned(), "bar".to_owned()], 1).unwrap(), Counter { count: 66 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 2);

            assert_eq!(dummy(&["foo".to_owned(), "foo".to_owned()], 1).unwrap(), Counter { count: 12 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 2);

            assert_eq!(dummy(&["foo".to_owned(), "bar".to_owned()], 11).unwrap(), Counter { count: 42 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 2);

            assert_eq!(dummy(&["foo".to_owned(), "bar".to_owned()], 2).unwrap(), Counter { count: 33 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 2);
        }

        #[test]
        fn test_agg_join_keys() {
            // Test that we get the correct result for each join
            // keys... and that we scan the data set only once.
            LOAD_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);

            assert_eq!(dummy(&["foo".to_owned()], 0).unwrap(), Counter { count: 0 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 1);

            assert_eq!(dummy(&["foo".to_owned()], 1).unwrap(), Counter { count: 6 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 1);

            assert_eq!(dummy(&["foo".to_owned()], 2).unwrap(), Counter { count: 3 });
            assert_eq!(LOAD_COUNT.load(std::sync::atomic::Ordering::Relaxed), 1);
        }
    }
}
