pub mod bad_trie;
pub mod index;
pub mod reverser;

use reverser::Inverse;
use reverser::InverseContext;

pub trait Value:
    merge::Merge + std::hash::Hash + PartialEq + Eq + Default + Sync + Clone + std::fmt::Debug
{
    fn is_default(&self) -> bool {
        self == &Default::default()
    }
}

pub trait JoinKey {
    type Ret: AsRef<[u8]>;

    fn to_bytes(&self) -> Self::Ret;
}

pub trait JoinKeys {
    type Ret<'a>: Sync;

    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Self::Ret<'a>;
}

impl<T> JoinKeys for T
where
    T: JoinKey + 'static,
{
    type Ret<'a> = Inverse<'a, T>;
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Self::Ret<'a> {
        ctx.fake(self).unwrap()
    }
}

impl<T> JoinKeys for (T,)
where
    T: JoinKey + 'static,
{
    type Ret<'a> = (Inverse<'a, T>,);
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Self::Ret<'a> {
        (ctx.fake(&self.0).unwrap(),)
    }
}

impl<T0, T1> JoinKeys for (T0, T1)
where
    T0: JoinKey + 'static,
    T1: JoinKey + 'static,
{
    type Ret<'a> = (Inverse<'a, T0>, Inverse<'a, T1>);
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Self::Ret<'a> {
        (ctx.fake(&self.0).unwrap(), ctx.fake(&self.1).unwrap())
    }
}

impl<T, const COUNT: usize> JoinKeys for [T; COUNT]
where
    T: JoinKey + 'static,
{
    type Ret<'a> = [Inverse<'a, T>; COUNT];
    fn invert<'a>(&self, ctx: &mut InverseContext<'a>) -> Self::Ret<'a> {
        self.each_ref().map(|x| ctx.fake(x).unwrap())
    }
}

impl JoinKey for u8 {
    type Ret = [u8; 1];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl JoinKey for u16 {
    type Ret = [u8; 2];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl JoinKey for u32 {
    type Ret = [u8; 4];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl JoinKey for u64 {
    type Ret = [u8; 8];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

impl JoinKey for u128 {
    type Ret = [u8; 16];

    fn to_bytes(&self) -> Self::Ret {
        self.to_be_bytes()
    }
}

static STRING_PARAMS: std::sync::LazyLock<umash::Params> =
    std::sync::LazyLock::new(Default::default);

impl JoinKey for &str {
    type Ret = [u8; 16];

    fn to_bytes(&self) -> Self::Ret {
        let fprint = STRING_PARAMS.fingerprint(self);
        let hi = fprint.hash() as u128;
        let lo = fprint.secondary() as u128;
        ((hi << 64) | lo).to_be_bytes()
    }
}
