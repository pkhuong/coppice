mod bad_trie;
mod map_reduce;
mod reverser;

use reverser::Inverse;
use reverser::InverseContext;

pub use map_reduce::clear_all_caches;
pub use map_reduce::map_reduce;

pub trait Aggregate:
    merge::Merge + std::hash::Hash + PartialEq + Eq + Default + Sync + Clone + std::fmt::Debug
{
    fn is_default(&self) -> bool {
        self == &Default::default()
    }
}

pub trait BaseJoinKey {
    type Ret: AsRef<[u8]>;

    fn to_bytes(&self) -> Self::Ret;
}

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
