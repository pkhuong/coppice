//! Coppice is a dynamic programming library for acyclic analytical
//! queries expressed as nested map/reduce computations over the union
//! of smaller data sets (tablets).  These map/reduce computations are
//! automatically cached and parallelised when executed with the
//! [`map_reduce()`] higher-order function.
//!
//! Of course, since we know we're only interested in final
//! [`map_reduce()`] results, we actually memoise fully aggregated
//! results for each "tablet" (small set of rows) and opaque `params`.
//!
//! In addition to these standard inputs (and cache keys), Coppice
//! also passes "join keys" to the mapped functions.  This third type
//! of inputs (in addition to rows and opaque `params`) enables
//! Coppice to offer asymptotically superior performance compared to
//! pure memoisation: [`map_reduce()`] essentially executes mapped
//! functions "in reverse," for join keys.
//!
//! The [`map_reduce()`] function ends up evaluating the mapped
//! function for each row and `params`, but extracts all join keys
//! that, combined with the row and `params`, yields a non-trivial
//! [`Aggregate`].  We thus extract, for each row and `params`, a
//! branching function from join keys to [`Aggregate`], and reduce
//! ([`merge::Merge`]) these branching functions together for all rows
//! in a tablet.  The maximum number of join keys that yield
//! non-trivial results for a given row (should) depend on the the
//! mapped function, but not on the rows or `params`... i.e., it's
//! a constant.
//!
//! For analytical queries, the number of scan over data files is
//! often what we really care about.  Pure memoisation gives us scans
//! proportional to `|tablets| * |params| * |join_keys|`; that's often
//! unrealistically large, which forces people to come up with ad hoc
//! indexing schemes.  The way Coppice caches branching functions
//! instead of raw values means the number of scans instead scales
//! with `|tablets| * |params|`.  When the join keys have a large
//! cardinality, shaving that `|join_keys|` multiplicative factor in
//! I/O can be a real practical win... hopefully enough to justify
//! the CPU overhead of Coppice's function inversion approach.
//!
//! Two key ideas underlie Coppice.
//!
//! The first is that backtracking search over join keys represented
//! as bounded arrays of bits gives us enough finite domain powers to
//! weakly emulate logic programming.  That's not a lot, but enough to
//! automatically build a bit-trie index from an opaque function.
//!
//! The second is that many analytical queries use only
//! [hierarchical joins](https://arxiv.org/abs/2201.05129) (a well known fact),
//! and that writing these queries as regular code implicitly gives us
//! the tree necessary for [Yannakakis's algorithm](https://en.wikipedia.org/wiki/Yannakakis_algorithm)
//! (maybe less well known).
//!
//! In short, the Coppice is just an API trick to coerce Rust coders
//! into writing plain code that can be executed with a version of
//! Yannakakis's algorithm simplified for the hierarchical subset of
//! acyclic queries.
//!
//! # Examples
//!
//! See `examples/ny_philharmonic.rs` for an executable example that
//! processes sets of JSON files that each contains metadata for the
//! New York Philharmonic Orchestra's performances over different
//! periods.
//!
//! Counting the total number of programs is a simple map/reduce function:
//!
//! ```ignore
//! fn load_json_dump(path: impl AsRef<Path>) -> Result<Vec<Program>, &'static str>;
//!
//! fn count_programs(files: &[PathBuf]) -> Result<u64, &'static str> {
//!     // Create a query cache where each dataset is identified by a PathBuf,
//!     // and the summary is just a counter
//!     coppice::query!(
//!         CACHE(path: PathBuf) -> Counter,
//!         load_json_dump(path),  // Load the data in each path with `load_json_dump`
//!         _program => Counter::new(1) // And count 1 for each Program in the json dump
//!     );
//!
//!     Ok(CACHE.nullary_query(files)?.count)
//! }
//! ```
//!
//! Counting the number of performances for each composer isn't hard either
//!
//! ```ignore
//! fn count_composer_occurrences(files: &[PathBuf]) -> Result<Vec<(String, u64)>, &'static str> {
//!     coppice::query!(
//!         // This time, we go from data in PathBuf to a Histogram keyed on String names
//!         CACHE(path: PathBuf) -> Histogram<String>,
//!         load_json_dump(path),  // Again, load the Programs in each json file
//!         row => {
//!             let mut ret: Histogram<String> = Default::default();
//!
//!             // For each work in the row (in the program), add 1
//!             // for each occurrence of a composer.
//!             for work in row.works.iter() {
//!                 if let Some(composer) = &work.composer_name {
//!                     ret.observe(composer.to_owned(), Counter::new(1));
//!                 }
//!             }
//!
//!             ret
//!         }
//!     );
//!
//!     Ok(CACHE.nullary_query(files)?.into_popularity_sorted_vec())
//! }
//! ```
//!
//! It's nice that the above is automatically cached and parallelised,
//! but that's nothing super interesting.  The next one should better
//! motivate the approach: we filter down the programs to only those that
//! occurred in a given venue, and accept an optional "root"
//! composer. The histogram count composer occurrences for programs
//! that included a given venue and in which the root composer was
//! also featured.
//!
//! ```ignore
//! fn count_composer_cooccurrences(
//!     files: &[PathBuf],
//!     venue: String,
//!     root_composer: Option<String>,
//! ) -> Result<Vec<(String, u64)>, &'static str> {
//!     coppice::query!(
//!         // We take a PathBuf, a venue, and maybe a root composer, and return a histogram keyed on composer names.
//!         COOCCURRENCES(path: PathBuf, venue: +String, root_composer: -Option<String>) -> Histogram<String>,
//!         load_json_dump(path),  // Again, load each `PathBuf` with `load_json_dump`.
//!         rows => {
//!             use rayon::iter::IntoParallelIterator;
//!             use rayon::iter::ParallelIterator;
//!
//!             let venue = venue.clone();  // Post-process the `Vec<Program>` returned by load_json_dump
//!             Ok(rows
//!                 .into_par_iter()
//!                 // Make sure the target venue appears in at least one of the concerts
//!                 .filter(move |row| row.concerts.iter().any(|concert| concert.venue == venue))
//!                 // extract the composer names.
//!                 .map(|row| {
//!                     row.works
//!                         .iter()
//!                         .map(|work| work.composer_name.clone())
//!                         .collect::<Vec<Option<String>>>()
//!                 }))
//!         },
//!         token, composers => {
//!             let _ = venue;  // We don't use the venue here, it was already handled above
//!             let mut ret: Histogram<String> = Default::default();
//!
//!             let mut maybe_composers: Vec<&Option<String>> = vec![&None];
//!             maybe_composers.extend(composers.iter());
//!
//!             let (mut token, root_composer) = token.focus(root_composer);
//!
//!             // If either `root_composer` is None, or matches one of the composers
//!             // in the program...
//!             let any_match = token.eql(root_composer, &None) || composers.iter().any(|composer| token.eql(root_composer, composer));
//!
//!             if any_match {
//!                 // Count occurrences in the histogram
//!                 for composer in composers.iter().flatten().cloned() {
//!                     ret.observe(composer, Counter::new(1));
//!                 }
//!             }
//!
//!             ret
//!         }
//!     );
//!
//!     Ok(COOCCURRENCES
//!         .query(files, &venue, &root_composer)?
//!         .into_popularity_sorted_vec())
//! }
//! ```
//!
//! This more complex examples shows what's interesting about Coppice:
//! the `query` call scans the files *once* regardless of how many
//! different `root_composer` values we pass.
//!
//! On my laptop, the first call to `count_composer_cooccurrences` takes 2 seconds.
//! Subsequence calls with various root composers (e.g., count how many times works
//! by each composer was played in the same program as Wagner) take ~100 *micro* seconds,
//! without any file I/O.  This is possible because `COOCCURRENCES.query` enumerates
//! all possible values of `root_composer` that would result in a non-trival result
//! for each row, and caches the result in a (bad) trie.

pub mod aggregates;
mod bad_trie;
mod map_reduce;
mod reverser;

use reverser::Inverse;
use reverser::InverseContext;

pub use map_reduce::clear_all_caches;
pub use map_reduce::make_map_map_reduce;
pub use map_reduce::make_map_reduce;
pub use map_reduce::map_map_reduce;
pub use map_reduce::map_reduce;
pub use map_reduce::Query;

/// Builds a `static Box<dyn Query>` for a fixed set of loading / mapping functions.
///
/// The general form is
///
/// ```ignore
/// query!(QUERY_NAME(tablet_var: TabletType, params: +ParamsType, join_keys: -JoinKeys) -> Summary,
///       [load_tablet(tablet_var)],
///       loaded_data => [transform(params, loaded_data)],
///       token, row => [map_function(token, params, join_keys, row)]);
/// ```
///
/// The `loaded_data => [transform ...]` entry is optional: when missing, the
/// values returned by `load_tablet` are passed straight to the map function
/// (row by row).
///
/// Any or both the `-params: ParamsType` and `+join_keys: JoinKeys`
/// parameters may be omitted.  When absent, they're implicitly the
/// unit `()`.
#[macro_export]
macro_rules! query {
    // General form (map reduce / map map reduce)
    ($name:ident($load_arg:ident: $Tablet:ty, $params_arg:ident: +$Params:ty, $join_args:ident: -$JoinKeys:ty) -> $Summary:ty,
     $load_expr:expr,
     $token_arg:ident, $row_arg:ident => $row_expr:expr) => {
        static $name: std::sync::LazyLock<
            Box<dyn coppice::Query<$Tablet, $Params, $JoinKeys, $Summary>>,
        > = std::sync::LazyLock::new(|| {
            coppice::make_map_reduce::<$Summary, $Tablet, $Params, $JoinKeys, _, _, _, _>(
                &|$load_arg| $load_expr,
                &|$token_arg, $params_arg, $join_args, $row_arg| $row_expr,
            )
        });
    };

    ($name:ident($load_arg:ident: $Tablet:ty, $params_arg:ident: +$Params:ty, $join_args:ident: -$JoinKeys:ty) -> $Summary:ty,
     $load_expr:expr,
     $input_arg:ident => $transform_expr:expr,
     $token_arg:ident, $row_arg:ident => $row_expr:expr) => {
        static $name: std::sync::LazyLock<
            Box<dyn coppice::Query<$Tablet, $Params, $JoinKeys, $Summary>>,
        > = std::sync::LazyLock::new(|| {
            coppice::make_map_map_reduce::<$Summary, $Tablet, $Params, $JoinKeys, _, _, _, _, _, _>(
                &|$load_arg| $load_expr,
                &|$params_arg, $input_arg| $transform_expr,
                &|$token_arg, $params_arg, $join_args, $row_arg| $row_expr,
            )
        });
    };

    // ParamQuery
    ($name:ident($load_arg:ident: $Tablet:ty, $params_arg:ident: +$Params:ty) -> $Summary:ty,
     $load_expr:expr,
     $token_arg:ident, $row_arg:ident => $row_expr:expr) => {
        static $name: std::sync::LazyLock<
            Box<dyn coppice::ParamQuery<$Tablet, $Params, $Summary>>,
        > = std::sync::LazyLock::new(|| {
            coppice::make_map_reduce::<$Summary, $Tablet, $Params, (), _, _, _, _>(
                &|$load_arg| $load_expr,
                &|$token_arg, $params_arg, _, $row_arg| $row_expr,
            )
        });
    };

    ($name:ident($load_arg:ident: $Tablet:ty, $params_arg:ident: +$Params:ty) -> $Summary:ty,
     $load_expr:expr,
     $input_arg:ident => $transform_expr:expr,
     $token_arg:ident, $row_arg:ident => $row_expr:expr) => {
        static $name: std::sync::LazyLock<
            Box<dyn coppice::ParamQuery<$Tablet, $Params, $Summary>>,
        > = std::sync::LazyLock::new(|| {
            coppice::make_map_map_reduce::<$Summary, $Tablet, $Params, (), _, _, _, _, _, _>(
                &|$load_arg| $load_expr,
                &|$params_arg, $input_arg| $transform_expr,
                &|$token_arg, $params_arg, _, $row_arg| $row_expr,
            )
        });
    };

    // JoinQuery
    ($name:ident($load_arg:ident: $Tablet:ty, $join_args:ident: -$JoinKeys:ty) -> $Summary:ty,
     $load_expr:expr,
     $token_arg:ident, $row_arg:ident => $row_expr:expr) => {
        static $name: std::sync::LazyLock<
            Box<dyn coppice::JoinQuery<$Tablet, $JoinKeys, $Summary>>,
        > = std::sync::LazyLock::new(|| {
            coppice::make_map_reduce::<$Summary, $Tablet, (), $JoinKeys, _, _, _, _>(
                &|$load_arg| $load_expr,
                &|$token_arg, _, $join_args, $row_arg| $row_expr,
            )
        });
    };

    ($name:ident($load_arg:ident: $Tablet:ty, $join_args:ident: -$JoinKeys:ty) -> $Summary:ty,
     $load_expr:expr,
     $input_arg:ident => $transform_expr:expr,
     $token_arg:ident, $row_arg:ident => $row_expr:expr) => {
        static $name: std::sync::LazyLock<
            Box<dyn coppice::JoinQuery<$Tablet, $JoinKeys, $Summary>>,
        > = std::sync::LazyLock::new(|| {
            coppice::make_map_map_reduce::<$Summary, $Tablet, (), $JoinKeys, _, _, _, _, _, _>(
                &|$load_arg| $load_expr,
                &|_, $input_arg| $transform_expr,
                &|$token_arg, _, $join_args, $row_arg| $row_expr,
            )
        });
    };

    // NullaryQuery
    ($name:ident($load_arg:ident: $Tablet:ty) -> $Summary:ty,
     $load_expr:expr,
     $row_arg:ident => $row_expr:expr
    ) => {
        static $name: std::sync::LazyLock<Box<dyn coppice::NullaryQuery<$Tablet, $Summary>>> =
            std::sync::LazyLock::new(|| {
                coppice::make_map_reduce::<$Summary, $Tablet, (), (), _, _, _, _>(
                    &|$load_arg| $load_expr,
                    &|_, _, _, $row_arg| $row_expr,
                )
            });
    };

    ($name:ident($load_arg:ident:$Tablet:ty) -> $Summary:ty,
     $load_expr:expr,
     $input_arg:ident => $transform_expr:expr,
     $row_arg:ident => $row_expr:expr
    ) => {
        static $name: std::sync::LazyLock<Box<dyn coppice::NullaryQuery<$Tablet, $Summary>>> =
            std::sync::LazyLock::new(|| {
                coppice::make_map_map_reduce::<$Summary, $Tablet, (), (), _, _, _, _>(
                    &|$load_arg| $load_expr,
                    &|_, $input_arg| $transform_expr,
                    &|_, _, _, $row_arg| $row_expr,
                )
            });
    };
}

/// A [`NullaryQuery`] is a [`Query`] that accepts only a list of tablets:
/// the params and join keys are always ().
pub trait NullaryQuery<
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Summary: Aggregate + Send + 'static,
>: Query<Tablet, (), (), Summary>
{
    fn nullary_query(&self, tablets: &[Tablet]) -> Result<Summary, &'static str> {
        self.query(tablets, &(), &())
    }
}

impl<
        Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
        Summary: Aggregate + Send + 'static,
        T: Query<Tablet, (), (), Summary>,
    > NullaryQuery<Tablet, Summary> for T
{
}

/// A [`ParamQuery`] is a [`Query`] that accepts only a list of tablets
/// and query parameters; join keys are always ().
pub trait ParamQuery<
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Params: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    Summary: Aggregate + Send + 'static,
>: Query<Tablet, Params, (), Summary>
{
    fn param_query(&self, tablets: &[Tablet], params: &Params) -> Result<Summary, &'static str> {
        self.query(tablets, params, &())
    }
}

/// A [`JoinQuery`] is a query that accepts only a list of tablets
/// and join keys; query parameters are always ().
pub trait JoinQuery<
    Tablet: std::hash::Hash + Eq + Clone + Sync + Send + 'static,
    JoinKeysT: JoinKeys + ?Sized + 'static,
    Summary: Aggregate + Send + 'static,
>: Query<Tablet, (), JoinKeysT, Summary>
{
    fn join_query(
        &self,
        tablets: &[Tablet],
        join_keys: &JoinKeysT,
    ) -> Result<Summary, &'static str> {
        self.query(tablets, &(), join_keys)
    }
}

/// Coppice caches results from aggregate queries where the query results
/// implement the [`Aggregate`] trait.
///
/// The [`merge::Merge`] trait must implement a commutative and associative
/// (abelian group) operator (e.g., sum for a counter, mininum or maximum for
/// watermarks).  We also assume that the default value is the identify (i.e.,
/// merging the default value with anything is a no-op).
///
///
/// There's a lot of requirements on that trait, but it's hard to see how to do
/// without most of them. We need:
///
/// - [`std::hash::Hash`] / [`PartialEq`] / [`Eq`] for hash consing
/// - [`Default`] to present a consistent interface when merging the empty set
/// - [`merge::Merge`] isn't mandatory, but it's nice to benefit from the derive macro
/// - [`Clone`] is needed because [`merge::Merge::merge`] consumes the "other" value
/// - [`Sync`] for parallelism
pub trait Aggregate:
    merge::Merge + std::hash::Hash + PartialEq + Eq + Default + Sync + Clone
{
    type Inner;

    fn is_neutral(&self) -> bool {
        self == &Default::default()
    }

    fn into_inner(self) -> Self::Inner;
}

/// In most cases (e.g., standard and automatically derived Hash
/// traits), the hash trait feeds different bytes for different
/// inputs.  In that case, we can compute a fingerprint with the
/// regular [`std::hash::Hash`] implementation.
pub trait HashIsInjective: std::hash::Hash + Eq {}

/// Individual join keys (that are run "in reverse") must be convertible to byte
/// arrays (bit vectors really, but byte arrays are convenient).
pub trait BaseJoinKey {
    type Ret: AsRef<[u8]>;

    fn to_bytes(&self) -> Self::Ret;
}

/// In practice, join keys are usually passed around as tuples or arrays, and
/// must be convertible to a.  The [`JoinKeys`] trait captures the containers of
/// [`BaseJoinKey`]s we know how to convert to function inversion input.
pub trait JoinKeys {
    type Ret<'a>: Sync;

    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str>;
}

impl<T> JoinKeys for T
where
    T: BaseJoinKey + ?Sized + 'static,
{
    type Ret<'a> = Inverse<'a, T>;
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str> {
        ctx.fake(self)
    }
}

impl JoinKeys for () {
    type Ret<'a> = ();
    fn invert<'a>(&self, _ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str> {
        Ok(())
    }
}

impl<T> JoinKeys for (T,)
where
    T: JoinKeys,
{
    type Ret<'a> = (T::Ret<'a>,);
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str> {
        Ok((self.0.invert(ctx)?,))
    }
}

impl<T0, T1> JoinKeys for (T0, T1)
where
    T0: JoinKeys,
    T1: JoinKeys,
{
    type Ret<'a> = (T0::Ret<'a>, T1::Ret<'a>);
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str> {
        Ok((self.0.invert(ctx)?, self.1.invert(ctx)?))
    }
}

impl<T0: JoinKeys, T1: JoinKeys, T2: JoinKeys> JoinKeys for (T0, T1, T2) {
    type Ret<'a> = (T0::Ret<'a>, T1::Ret<'a>, T2::Ret<'a>);
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str> {
        Ok((
            self.0.invert(ctx)?,
            self.1.invert(ctx)?,
            self.2.invert(ctx)?,
        ))
    }
}

impl<T0: JoinKeys, T1: JoinKeys, T2: JoinKeys, T3: JoinKeys> JoinKeys for (T0, T1, T2, T3) {
    type Ret<'a> = (T0::Ret<'a>, T1::Ret<'a>, T2::Ret<'a>, T3::Ret<'a>);
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str> {
        Ok((
            self.0.invert(ctx)?,
            self.1.invert(ctx)?,
            self.2.invert(ctx)?,
            self.3.invert(ctx)?,
        ))
    }
}

impl<T0: JoinKeys, T1: JoinKeys, T2: JoinKeys, T3: JoinKeys, T4: JoinKeys> JoinKeys
    for (T0, T1, T2, T3, T4)
{
    type Ret<'a> = (
        T0::Ret<'a>,
        T1::Ret<'a>,
        T2::Ret<'a>,
        T3::Ret<'a>,
        T4::Ret<'a>,
    );
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str> {
        Ok((
            self.0.invert(ctx)?,
            self.1.invert(ctx)?,
            self.2.invert(ctx)?,
            self.3.invert(ctx)?,
            self.4.invert(ctx)?,
        ))
    }
}

impl<T, const COUNT: usize> JoinKeys for [T; COUNT]
where
    T: JoinKeys,
{
    type Ret<'a> = [T::Ret<'a>; COUNT];
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Result<Self::Ret<'a>, &'static str> {
        let mut err: Option<&'static str> = None;
        let ret = self.each_ref().map(|x| match x.invert(ctx) {
            Ok(inv) => Some(inv),
            Err(e) => {
                err = Some(e);
                None
            }
        });

        match err {
            Some(e) => Err(e),
            None => Ok(ret.map(|x| x.unwrap())),
        }
    }
}

impl BaseJoinKey for u8 {
    type Ret = [u8; 1];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl BaseJoinKey for u16 {
    type Ret = [u8; 2];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl BaseJoinKey for u32 {
    type Ret = [u8; 4];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl BaseJoinKey for u64 {
    type Ret = [u8; 8];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl BaseJoinKey for u128 {
    type Ret = [u8; 16];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl HashIsInjective for str {}
impl HashIsInjective for String {}
impl<T: HashIsInjective> HashIsInjective for Option<T> {}
impl<T: HashIsInjective> HashIsInjective for Vec<T> {}
impl<T: HashIsInjective> HashIsInjective for [T] {}

static INJECTIVE_HASH_PARAMS: std::sync::LazyLock<umash::Params> =
    std::sync::LazyLock::new(Default::default);

impl<T: HashIsInjective> BaseJoinKey for T {
    type Ret = [u8; 16];

    fn to_bytes(&self) -> Self::Ret {
        let fprint = INJECTIVE_HASH_PARAMS.fingerprint(self);
        let hi = fprint.hash() as u128;
        let lo = fprint.secondary() as u128;
        ((hi << 64) | lo).to_be_bytes()
    }
}
