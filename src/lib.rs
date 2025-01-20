pub mod bad_trie;
pub mod reverser;

pub trait Value:
    merge::Merge + std::hash::Hash + PartialEq + Eq + Default + Sync + Clone + std::fmt::Debug
{
    fn is_default(&self) -> bool {
        self == &Default::default()
    }
}
