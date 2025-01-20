//! Coppice is a dynamic programming library for acyclic analytical
//! queries expressed as nested map/reduce computations.  These
//! map/reduce computations are automatically cached and parallelised
//! when executed with the [`map_reduce()`] higher-order function.
//!
//! Of course, since we know we're only interested in full blown
//! [`map_reduce()`] results, we actually memoise fully aggregated
//! results for each "tablet" (small set of rows) and opaque `params`.
//!
//! In addition to these standard inputs (and cache keys), Coppice
//! also passes "join keys" to the "map" functions.  This third type
//! of inputs (in addition to rows and opaque `params`) enables
//! Coppice to offer asymptotically superior performance compared to
//! pure memoisation: [`map_reduce()`] essentially executes "map"
//! functions "in reverse," for join keys.
//!
//! The [`map_reduce()`] function thus ends up evaluating the "map"
//! function for each row and `params`, but extracts all join keys
//! that, combined with the row and `params`, yields a non-trivial
//! [`Aggregate`].  We thus extract, for each row and `params`,
//! a branching function from join keys to [`Aggregate`], and reduce
//! ([`merge::Merge`]) these branching functions together for
//! all rows in a tablet.
//!
//! Two key insights underlie Coppice.
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
//! In short, the Coppice is just an API trick to coerce Rust covers
//! into writing plain code that can be executed with a version of
//! Yannakakis's algorithm simplified for the hierarchical subset of
//! acyclic queries.
mod bad_trie;
mod map_reduce;
mod reverser;

use reverser::Inverse;
use reverser::InverseContext;

pub use map_reduce::clear_all_caches;
pub use map_reduce::map_reduce;

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
    fn is_default(&self) -> bool {
        self == &Default::default()
    }
}

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
    T: BaseJoinKey + 'static,
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

static STRING_PARAMS: std::sync::LazyLock<umash::Params> =
    std::sync::LazyLock::new(Default::default);

impl BaseJoinKey for &str {
    type Ret = [u8; 16];

    fn to_bytes(&self) -> Self::Ret {
        let fprint = STRING_PARAMS.fingerprint(self);
        let hi = fprint.hash() as u128;
        let lo = fprint.secondary() as u128;
        ((hi << 64) | lo).to_be_bytes()
    }
}
