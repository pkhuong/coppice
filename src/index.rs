//! Index data structure based on running queries in reverse.
use std::collections::HashMap;
use std::sync::Arc;

use crate::bad_trie::Node;
use crate::reverser::InverseContext;
use crate::reverser::SearchToken;
use crate::Value;

struct Index<Tablet: std::hash::Hash + Eq, Summary: Value> {
    inverted_tablets: HashMap<Tablet, Node<Summary>>,
}

pub fn aggregate<Summary, Tablet, Params, Row, Rows, JoinKeys, JoinKeyFn, RowFn, WorkerFn>(
    tablet: Tablet,
    params: Params,
    join_key_fn: JoinKeyFn,
    row_fn: RowFn,
    worker: WorkerFn,
) -> Result<Option<Arc<Summary>>, &'static str>
where
    Summary: Value + Send,
    Tablet: std::hash::Hash + Eq,
    Params: Sync,
    Row: Send,
    Rows: rayon::iter::IntoParallelIterator<Item = Row>,
    JoinKeys: Sync,
    JoinKeyFn: for<'a> FnOnce(&mut InverseContext<'a>) -> JoinKeys,
    RowFn: FnOnce(&Tablet) -> Rows,
    WorkerFn:
        for<'a> Fn(SearchToken<'a>, &JoinKeys, &Row) -> (SearchToken<'a>, Summary) + Sync + Send,
{
    let builder = crate::bad_trie::Builder::new();
    let mut ctx = InverseContext::new();
    let join_keys = join_key_fn(&mut ctx);

    let cache = crate::reverser::map_reverse(&builder, row_fn(&tablet), &join_keys, worker)?;
    cache.lookup(&ctx.keys())
}
