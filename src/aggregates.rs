//! Common aggregation results.
use crate::Aggregate;

use std::collections::BTreeMap;
use std::hash::Hash;

use merge::Merge;

/// A [`Counter`] sums values.
#[derive(Clone, Copy, Default, Debug, Eq, Hash, PartialEq)]
pub struct Counter {
    pub count: u64,
}

impl Counter {
    /// Returns a fresh counter initialised at `count`.
    pub fn new(count: u64) -> Counter {
        Counter { count }
    }
}

impl Merge for Counter {
    fn merge(&mut self, other: Counter) {
        self.count += other.count;
    }
}

impl Aggregate for Counter {
    type Inner = u64;

    fn into_inner(self) -> u64 {
        self.count
    }
}

/// A [`KeyedAggregate`] merges values associated with the same key.
#[derive(Clone, Eq, std::hash::Hash, PartialEq)]
pub struct KeyedAggregate<Key: Clone + Ord + Hash + Sync, Value: Aggregate> {
    pub values: std::collections::BTreeMap<Key, Value>,
}

impl<K: Clone + Ord + Hash + Sync, V: Aggregate> Aggregate for KeyedAggregate<K, V> {
    type Inner = BTreeMap<K, V::Inner>;

    fn into_inner(self) -> Self::Inner {
        self.values
            .into_iter()
            .map(|(k, v)| (k, v.into_inner()))
            .collect::<_>()
    }
}

impl<K: Clone + Ord + Hash + Sync, V: Aggregate> KeyedAggregate<K, V> {
    /// Returns a fresh `KeyedAggregate` with a single key-value pair.
    pub fn new(key: K, value: V) -> Self {
        let mut ret = Self {
            values: Default::default(),
        };

        ret.values.insert(key, value);
        ret
    }

    /// Updates this `KeyedAggregate` to set/augment `value` for `key`.
    pub fn observe(&mut self, key: K, value: V) {
        use std::collections::btree_map::Entry;

        match self.values.entry(key) {
            Entry::Occupied(ref mut entry) => {
                entry.get_mut().merge(value);
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
            }
        };
    }
}

impl<K: Clone + Ord + Hash + Sync, V: Aggregate> KeyedAggregate<K, V>
where
    V::Inner: Ord,
{
    /// Returns the contents, sorted by the value in descending order.
    pub fn into_popularity_sorted_vec(self) -> Vec<(K, V::Inner)> {
        let mut ret = self
            .values
            .into_iter()
            .map(|(k, v)| (k, v.into_inner()))
            .collect::<Vec<_>>();
        ret.sort_by(|x, y| x.1.cmp(&y.1).reverse());
        ret
    }
}

impl<K: Clone + Ord + Hash + Sync, V: Aggregate> Default for KeyedAggregate<K, V> {
    fn default() -> Self {
        Self {
            values: Default::default(),
        }
    }
}

impl<K: Clone + Ord + Hash + Sync, V: Aggregate> Merge for KeyedAggregate<K, V> {
    fn merge(&mut self, mut other: KeyedAggregate<K, V>) {
        if self.values.len() < other.values.len() {
            std::mem::swap(self, &mut other);
        }
        for (key, value) in other.values.into_iter() {
            self.observe(key, value);
        }
    }
}

/// A [`Histogram`] is a [`KeyedAggregate`] from arbitrary key to (occurrence) [`Counter`].
pub type Histogram<T> = KeyedAggregate<T, Counter>;
