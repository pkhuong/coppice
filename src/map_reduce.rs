//! Index data structure based on running queries in reverse.
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use quinine::MonoBox;

use crate::bad_trie;
use crate::reverser::InverseContext;
use crate::reverser::SearchToken;
use crate::Aggregate;
use crate::JoinKeys;

/// A [`Query`] object accepts `Tablet`s, `Params`, and `JoinKeysT`,
/// and returns a `Summary` for the union of all the `Tablets`, given
/// the `Params`, and `JoinKeysT`.
///
/// The actual data loading and processing logic is specified when the
/// [`Query`] object is created.
pub trait Query<
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    Summary: Aggregate + Send + 'static,
>: Sync + Send
{
    /// Loads data associated with `tablets`, and processes it
    /// according to the `params` and `join_keys`.
    ///
    /// The `tablets` and `params` serve as toplevel cache key:
    /// we'll load the data associated with `tablets` for each
    /// fresh `Tablet x Params` seen.
    ///
    /// The `join_keys`, on the order hand, are enumerated ahead of
    /// time and fresh `join_keys` will thus not trigger extra I/O.
    fn query(
        &self,
        tablets: &[Tablet],
        params: &Params,
        join_keys: &JoinKeysT,
    ) -> Result<Summary, &'static str>;

    /// Dissociates the [`Query`] object from its current result
    /// cache, and re-associates it with a fresh empty cache.
    ///
    /// This frees up memory, but may take a while, and obviously
    /// means future queries will find a cold cache.
    ///
    /// The return value lets the caller delay destruction until
    /// the call has returned.
    fn reset_cache(&mut self) -> Arc<dyn std::any::Any + Send>;
}

// A NodeCell is a future implemented as a Box and a worker lock.
type NodeCell<T> = Arc<(MonoBox<bad_trie::Node<T>>, Mutex<()>)>;

// Shared mutable state for queries: the trie builder and
// the actual index, from (Tablet, Param) => bit trie.
struct SummaryCache<
    Tablet: std::hash::Hash + Eq,
    Params: std::hash::Hash + Eq + Clone,
    Summary: Aggregate,
> {
    builder: bad_trie::Builder<Summary>,
    inverted_summaries: Mutex<HashMap<(Tablet, Params), NodeCell<Summary>>>,
}

impl<Tablet: std::hash::Hash + Eq, Params: std::hash::Hash + Eq + Clone, Summary: Aggregate>
    SummaryCache<Tablet, Params, Summary>
{
    pub fn new() -> Self {
        SummaryCache {
            builder: Default::default(),
            inverted_summaries: Mutex::new(Default::default()),
        }
    }

    #[inline]
    pub fn get_all_cells(&self, keys: Vec<Tablet>, params: &Params) -> Vec<NodeCell<Summary>> {
        let mut tablets = self.inverted_summaries.lock().unwrap();

        keys.into_iter()
            .map(|tablet| tablets.entry((tablet, params.clone())).or_default().clone())
            .collect::<Vec<_>>()
    }
}

// We cache `SUMMARY_CACHES` to maximise sharing.  Unfortunately, this
// also means that we'll hang on to results longer than we might want.
type ShapeKey = (
    std::any::TypeId, // Summary
    std::any::TypeId, // Tablet
    std::any::TypeId, // Params
    std::any::TypeId, // JoinKeys
    std::any::TypeId, // RowFn
    std::any::TypeId, // TransformFn
    std::any::TypeId, // WorkerFn
    usize,            // &RowFn
    usize,            // &TransformFn
    usize,            // &WorkerFn
);

static SUMMARY_CACHES: std::sync::LazyLock<
    Mutex<HashMap<ShapeKey, Arc<dyn std::any::Any + Sync + Send>>>,
> = std::sync::LazyLock::new(Default::default);

/// Clears *all* query caches for [`map_reduce`] or [`map_map_reduce`]
/// calls in the current process.
pub fn clear_all_caches() {
    SUMMARY_CACHES.lock().unwrap().clear();
}

/// The only implementation of [`Query`]
struct MapReduceCache<
    Tablet,
    Params,
    JoinKeysT,
    Summary,
    InputRows,
    RowFn,
    TransformFn,
    TransformedRow,
    TransformedRows,
    WorkerFn,
> where
    Summary: Aggregate + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    RowFn: Fn(&Tablet) -> Result<InputRows, &'static str> + Sync + 'static,

    TransformedRow: Send,
    TransformedRows: rayon::iter::IntoParallelIterator<Item = TransformedRow>,
    TransformFn: Fn(&Params, InputRows) -> Result<TransformedRows, &'static str> + Sync + 'static,

    WorkerFn: for<'a, 'b> Fn(
            SearchToken<'a, 'b>,
            &Params,
            &JoinKeysT::Ret<'a>,
            &TransformedRow,
        ) -> Summary
        + Sync
        + 'static,
{
    cache: Arc<SummaryCache<Tablet, Params, Summary>>,
    row_fn: &'static RowFn,
    transform_fn: &'static TransformFn,
    worker: &'static WorkerFn,

    _marker: std::marker::PhantomData<fn(&JoinKeysT) -> ()>,
}

impl<
        Tablet,
        Params,
        JoinKeysT,
        Summary,
        InputRows,
        RowFn,
        TransformFn,
        TransformedRow,
        TransformedRows,
        WorkerFn,
    >
    MapReduceCache<
        Tablet,
        Params,
        JoinKeysT,
        Summary,
        InputRows,
        RowFn,
        TransformFn,
        TransformedRow,
        TransformedRows,
        WorkerFn,
    >
where
    Summary: Aggregate + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    RowFn: Fn(&Tablet) -> Result<InputRows, &'static str> + Sync + 'static,

    TransformedRow: Send,
    TransformedRows: rayon::iter::IntoParallelIterator<Item = TransformedRow>,
    TransformFn: Fn(&Params, InputRows) -> Result<TransformedRows, &'static str> + Sync + 'static,

    WorkerFn: for<'a, 'b> Fn(
            SearchToken<'a, 'b>,
            &Params,
            &JoinKeysT::Ret<'a>,
            &TransformedRow,
        ) -> Summary
        + Sync
        + 'static,
{
    // The `new` function is responsible for checking parametricity
    // constraints.  Once a [`MapReduceCache`] has been constructed,
    // we assume everything is consistent.
    #[inline(never)]
    fn new(
        row_fn: &'static RowFn,
        transform_fn: &'static TransformFn,
        worker: &'static WorkerFn,
    ) -> Self {
        use std::any::TypeId;

        let shape_key: ShapeKey = (
            TypeId::of::<Summary>(),
            TypeId::of::<Tablet>(),
            TypeId::of::<Params>(),
            TypeId::of::<JoinKeysT>(),
            TypeId::of::<RowFn>(),
            TypeId::of::<TransformFn>(),
            TypeId::of::<WorkerFn>(),
            row_fn as *const RowFn as usize,
            transform_fn as *const TransformFn as usize,
            worker as *const WorkerFn as usize,
        );
        let cache = SUMMARY_CACHES
            .lock()
            .unwrap()
            .entry(shape_key)
            .or_insert_with(|| Arc::new(SummaryCache::<Tablet, Params, Summary>::new()))
            .clone();

        let cache = cache
            .downcast::<SummaryCache<Tablet, Params, Summary>>()
            .expect("types are part of the shape key");

        Self {
            cache,
            row_fn,
            transform_fn,
            worker,
            _marker: Default::default(),
        }
    }

    #[inline(never)]
    fn query_impl(
        &self,
        tablets: &[Tablet],
        params: &Params,
        join_keys: &JoinKeysT::Ret<'static>,
    ) -> Result<Vec<NodeCell<Summary>>, &'static str> {
        let node_cells: Vec<NodeCell<Summary>> = self.cache.get_all_cells(tablets.to_vec(), params);

        let row_fn = self.row_fn;
        let transform_fn = self.transform_fn;
        let worker = self.worker;
        // We use this array of flags to ensure we spawn each work unit at most once
        // (modulo poisoning).
        let mut already_spawned_flag: Vec<bool> = vec![false; node_cells.len()];
        let builder = &self.cache.builder;
        let err: Mutex<Option<&'static str>> = Default::default();

        rayon::in_place_scope_fifo(|scope| {
            // Either spawn fresh work units to try and compute values,
            // or yield until they're all computed.
            loop {
                if err.lock().unwrap().is_some() {
                    break;
                }

                let mut all_done = true;
                for ((tablet, spawned_flag), node_cell) in tablets
                    .iter()
                    .zip(already_spawned_flag.iter_mut())
                    .zip(node_cells.iter())
                {
                    if node_cell.0.is_some() {
                        continue;
                    }

                    all_done = false;

                    if node_cell.1.is_poisoned() {
                        node_cell.1.clear_poison();
                        *spawned_flag = false;
                    }

                    if *spawned_flag {
                        continue;
                    }

                    // Someone's already working on it.
                    if node_cell.1.try_lock().is_err() {
                        continue;
                    }

                    *spawned_flag = true;
                    // This is the only interesting piece of code.
                    scope.spawn_fifo(|_scope| {
                        let Ok(_guard) = node_cell.1.try_lock() else {
                            return;
                        };

                        // Try to load the data corresponding to `tablet`.
                        let rows = match row_fn(tablet) {
                            Ok(rows) => rows,
                            Err(e) => {
                                err.lock().unwrap().get_or_insert(e);
                                return;
                            }
                        };

                        // Transform the data
                        let rows = match transform_fn(params, rows) {
                            Ok(rows) => rows,
                            Err(e) => {
                                err.lock().unwrap().get_or_insert(e);
                                return;
                            }
                        };

                        // Invert the worker function
                        let result = crate::reverser::map_reverse(
                            builder,
                            rows,
                            join_keys,
                            |token, join_keys, row| worker(token, params, join_keys, row),
                        );

                        // Stick the result in the `node_cell`.
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
}

impl<
        Tablet,
        Params,
        JoinKeysT,
        Summary,
        InputRows,
        RowFn,
        TransformFn,
        TransformedRow,
        TransformedRows,
        WorkerFn,
    > Query<Tablet, Params, JoinKeysT, Summary>
    for MapReduceCache<
        Tablet,
        Params,
        JoinKeysT,
        Summary,
        InputRows,
        RowFn,
        TransformFn,
        TransformedRow,
        TransformedRows,
        WorkerFn,
    >
where
    Summary: Aggregate + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    RowFn: Fn(&Tablet) -> Result<InputRows, &'static str> + Sync + 'static,

    TransformedRow: Send,
    TransformedRows: rayon::iter::IntoParallelIterator<Item = TransformedRow>,
    TransformFn: Fn(&Params, InputRows) -> Result<TransformedRows, &'static str> + Sync + 'static,

    WorkerFn: for<'a, 'b> Fn(
            SearchToken<'a, 'b>,
            &Params,
            &JoinKeysT::Ret<'a>,
            &TransformedRow,
        ) -> Summary
        + Sync
        + 'static,
{
    #[inline(never)]
    fn query(
        &self,
        tablets: &[Tablet],
        params: &Params,
        join_keys: &JoinKeysT,
    ) -> Result<Summary, &'static str> {
        let mut ctx = InverseContext::<'static>::new();
        let join_keys = join_keys.invert(&mut ctx)?;

        let node_cells = self.query_impl(tablets, params, &join_keys)?;
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

    #[inline(never)]
    fn reset_cache(&mut self) -> Arc<dyn std::any::Any + Send> {
        let mut empty: Arc<SummaryCache<Tablet, Params, Summary>> = Arc::new(SummaryCache {
            builder: Default::default(),
            inverted_summaries: Default::default(),
        });
        std::mem::swap(&mut self.cache, &mut empty);

        empty
    }
}

/// Returns a [`Query`] object for the map reduce computation that loads data
/// from `Tablet`s with `row_fn`, and feed each resulting row, along with the
/// parameters and join keys, to `worker`.
#[inline(always)]
pub fn make_map_reduce<Summary, Tablet, Params, JoinKeysT, Row, Rows, RowFn, WorkerFn>(
    row_fn: &'static RowFn,
    worker: &'static WorkerFn,
) -> Box<impl Query<Tablet, Params, JoinKeysT, Summary>>
where
    Summary: Aggregate + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Row: Send,
    Rows: rayon::iter::IntoParallelIterator<Item = Row>,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    RowFn: Fn(&Tablet) -> Result<Rows, &'static str> + Sync,
    WorkerFn:
        for<'a, 'b> Fn(SearchToken<'a, 'b>, &Params, &JoinKeysT::Ret<'a>, &Row) -> Summary + Sync,
{
    Box::new(MapReduceCache::new(row_fn, &|_params, x| Ok(x), worker))
}

/// Returns a [`Query`] object for the map reduce computation that
/// loads data from `Tablet`s with `row_fn`, transforms them with
/// `transform_fm`, and feed each transformed row, along with the
/// parameters and join keys, to `worker`.
#[inline(always)]
pub fn make_map_map_reduce<
    Summary,
    Tablet,
    Params,
    JoinKeysT,
    InputRows,
    RowFn,
    TransformFn,
    TransformedRow,
    TransformedRows,
    WorkerFn,
>(
    row_fn: &'static RowFn,
    transform_fn: &'static TransformFn,
    worker: &'static WorkerFn,
) -> Box<impl Query<Tablet, Params, JoinKeysT, Summary>>
where
    Summary: Aggregate + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    RowFn: Fn(&Tablet) -> Result<InputRows, &'static str> + Sync,
    TransformedRow: Send,
    TransformedRows: rayon::iter::IntoParallelIterator<Item = TransformedRow>,
    TransformFn: Fn(&Params, InputRows) -> Result<TransformedRows, &'static str> + Sync,
    WorkerFn: for<'a, 'b> Fn(
            SearchToken<'a, 'b>,
            &Params,
            &JoinKeysT::Ret<'a>,
            &TransformedRow,
        ) -> Summary
        + Sync,
{
    Box::new(MapReduceCache::new(row_fn, transform_fn, worker))
}

/// The [`map_reduce`] generic function takes a list of tablets
/// (fractional data sets), converts each to a parallel iterator of
/// rows with the `row_fn`, and merges the result of executing
/// `worker` with the `params` and `join_keys` on each `row` in all
/// the `tablets`.
///
/// In other words, each call to the [`map_reduce`] function implements
/// `tablets.flat_map(row_fn).map(|row| worker(params, join_keys, row)).reduce(|x, y| x.merge(y))`
/// The complexity stems from the fact that `map_reduce` internally caches
/// the results so as to avoid rescanning the tablets when only the `join_keys`
/// differ.
///
/// The difference between `params` and `join_keys` is that the
/// `params` piped through like opaque values to `worker`, and are
/// used only as cache keys.
///
/// The `join_keys`, on the other hand, are kept hidden from `worker`
/// when building the cache: the cache construction process computes
/// all possible values for the `join_keys` that could yield a
/// non-default [`Aggregate`].
///
/// For this function inversion process to work correctly, the `row_fn`
/// and `worker` functions must be pure functions.  The caching logic
/// uses these functions as part of the cache key, so they must have
/// static lifetime (their addresses turn into cache keys).
#[inline(always)]
pub fn map_reduce<Summary, Tablet, Params, Row, Rows, JoinKeysT, RowFn, WorkerFn>(
    tablets: &[Tablet],
    params: &Params,
    join_keys: &JoinKeysT,
    row_fn: &'static RowFn,
    worker: &'static WorkerFn,
) -> Result<Summary, &'static str>
where
    Summary: Aggregate + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Row: Send,
    Rows: rayon::iter::IntoParallelIterator<Item = Row>,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    RowFn: Fn(&Tablet) -> Result<Rows, &'static str> + Sync,
    WorkerFn:
        for<'a, 'b> Fn(SearchToken<'a, 'b>, &Params, &JoinKeysT::Ret<'a>, &Row) -> Summary + Sync,
{
    let query = make_map_reduce(row_fn, worker);
    query.query(tablets, params, join_keys)
}

/// The [`map_map_reduce`] generic function is like [`map_reduce`],
/// except that an additionan `transform_fn` is applied to the output
/// of the `row_fn` before inverting the `worker_fn`.
///
/// In traditional map-reduce, this intermediate step could be fused
/// in a single map stage.  However, it can sometimes be helpful to
/// transform the whole set of rows in bulk, and the `transform_fn`
/// can also improve efficiency because it only runs once instead of
/// re-executing from scratch for each iteration of the `worker_fn`
/// inversion algorithm.
#[inline(always)]
pub fn map_map_reduce<
    Summary,
    Tablet,
    Params,
    InputRows,
    JoinKeysT,
    RowFn,
    TransformFn,
    TransformedRow,
    TransformedRows,
    WorkerFn,
>(
    tablets: &[Tablet],
    params: &Params,
    join_keys: &JoinKeysT,
    row_fn: &'static RowFn,
    transform_fn: &'static TransformFn,
    worker: &'static WorkerFn,
) -> Result<Summary, &'static str>
where
    Summary: Aggregate + Send + 'static,
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    RowFn: Fn(&Tablet) -> Result<InputRows, &'static str> + Sync,

    TransformedRow: Send,
    TransformedRows: rayon::iter::IntoParallelIterator<Item = TransformedRow>,
    TransformFn: Fn(&Params, InputRows) -> Result<TransformedRows, &'static str> + Sync,

    WorkerFn: for<'a, 'b> Fn(
            SearchToken<'a, 'b>,
            &Params,
            &JoinKeysT::Ret<'a>,
            &TransformedRow,
        ) -> Summary
        + Sync,
{
    make_map_map_reduce(row_fn, transform_fn, worker).query(tablets, params, join_keys)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::aggregates::Counter;
    use std::sync::atomic::Ordering;

    static LOAD_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn dummy(tablets: &[String], join_key: u8) -> Result<Counter, &'static str> {
        map_reduce(
            tablets,
            &("test",),
            &join_key,
            &|tablet| {
                LOAD_COUNT.fetch_add(1, Ordering::Relaxed);

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
            &|token, params, needle, (key, value)| {
                assert_eq!(*params, ("test",));
                let (mut token, needle) = token.focus(needle);

                let count = if token.eql(needle, key) { *value } else { 0 };
                Counter::new(count as u64)
            },
        )
    }

    use rusty_fork::rusty_fork_test;
    rusty_fork_test! {
        #[test]
        fn test_map_reduce_smoke() {
            rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();

            LOAD_COUNT.store(0, Ordering::Relaxed);

            assert_eq!(dummy(&["foo".to_owned()], 1).unwrap(), Counter { count: 6 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 1);

            assert_eq!(dummy(&["bar".to_owned()], 1).unwrap(), Counter { count: 60 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 2);

            assert_eq!(dummy(&["foo".to_owned(), "bar".to_owned()], 1).unwrap(), Counter { count: 66 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 2);

            assert_eq!(dummy(&["foo".to_owned(), "foo".to_owned()], 1).unwrap(), Counter { count: 12 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 2);

            assert_eq!(dummy(&["foo".to_owned(), "bar".to_owned()], 11).unwrap(), Counter { count: 42 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 2);

            assert_eq!(dummy(&["foo".to_owned(), "bar".to_owned()], 2).unwrap(), Counter { count: 33 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 2);
        }

        #[test]
        fn test_map_reduce_join_keys() {
            // Test that we get the correct result for each join
            // keys... and that we scan the data set only once.
            LOAD_COUNT.store(0, Ordering::Relaxed);

            assert_eq!(dummy(&["foo".to_owned()], 0).unwrap(), Counter { count: 0 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 1);

            assert_eq!(dummy(&["foo".to_owned()], 1).unwrap(), Counter { count: 6 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 1);

            assert_eq!(dummy(&["foo".to_owned()], 2).unwrap(), Counter { count: 3 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn test_map_reduce_clear() {
            LOAD_COUNT.store(0, Ordering::Relaxed);

            // Cache works
            assert_eq!(dummy(&["foo".to_owned()], 0).unwrap(), Counter { count: 0 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 1);
            assert_eq!(dummy(&["foo".to_owned()], 0).unwrap(), Counter { count: 0 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 1);

            clear_all_caches();

            // Must repopulate cache
            assert_eq!(dummy(&["foo".to_owned()], 0).unwrap(), Counter { count: 0 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 2);
            assert_eq!(dummy(&["foo".to_owned()], 0).unwrap(), Counter { count: 0 });
            assert_eq!(LOAD_COUNT.load(Ordering::Relaxed), 2);
        }
    }
}
