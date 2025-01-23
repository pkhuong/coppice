//! A bad bit trie.
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

use crate::Aggregate;

#[derive(Debug)]
pub struct Split<T: Aggregate> {
    input_id: u8,
    index: u32,
    arcs: [Node<T>; 2],
}

#[derive(Debug)]
pub enum Node<T: Aggregate> {
    Neutral,
    Leaf(Arc<T>),
    Split(Arc<Split<T>>),
}

impl<T: Aggregate> Clone for Node<T> {
    fn clone(&self) -> Node<T> {
        match self {
            Node::Neutral => Node::Neutral,
            Node::Leaf(leaf) => Node::Leaf(leaf.clone()),
            Node::Split(split) => Node::Split(split.clone()),
        }
    }
}

type SplitKey = (u8, u32, usize, usize);

#[derive(Debug, Default)]
pub struct Builder<T: Aggregate> {
    leaf_cache: Mutex<HashSet<Arc<T>>>,
    split_cache: Mutex<HashMap<SplitKey, Arc<Split<T>>>>,
}

impl<T: Aggregate> Node<T> {
    fn as_usize(&self) -> usize {
        match self {
            Node::Neutral => 0,
            Node::Leaf(ref leaf) => Arc::as_ptr(leaf) as usize,
            Node::Split(ref split) => Arc::as_ptr(split) as usize,
        }
    }

    pub fn lookup(&self, inputs: &[&[u8]]) -> Result<Option<Arc<T>>, &'static str> {
        match self {
            Node::Neutral => Ok(None),
            Node::Leaf(leaf) => Ok(Some(leaf.clone())),
            Node::Split(split) => {
                let input = inputs
                    .get(split.input_id as usize)
                    .ok_or("Unknown input id")?;
                let byte_index = (split.index as usize) / 8;
                let bit_index = (split.index as usize) % 8;

                let byte = input.get(byte_index).ok_or("Out of bound index bit")?;
                let value = (byte >> bit_index) & 1;

                split.arcs[value as usize].lookup(inputs)
            }
        }
    }
}

impl<T: Aggregate> Builder<T> {
    pub fn make_neutral(&self) -> Node<T> {
        Node::Neutral
    }

    fn make_leaf(&self, value: T) -> Node<T> {
        if value.is_neutral() {
            return self.make_neutral();
        }

        let mut cache = self.leaf_cache.lock().unwrap();

        if let Some(hit) = cache.get(&value) {
            Node::Leaf(hit.clone())
        } else {
            let ret = Arc::new(value);
            cache.insert(ret.clone());
            Node::Leaf(ret)
        }
    }

    fn make_split(&self, input_id: u8, index: u32, arcs: [Node<T>; 2]) -> Node<T> {
        if arcs[0].as_usize() == arcs[1].as_usize() {
            return arcs[0].clone();
        }

        let key = (input_id, index, arcs[0].as_usize(), arcs[1].as_usize());
        let mut cache = self.split_cache.lock().unwrap();

        let split = cache.entry(key).or_insert_with(|| {
            Arc::new(Split {
                input_id,
                index,
                arcs,
            })
        });
        Node::Split(split.clone())
    }

    pub fn make_spine(&self, path: &[(u8, u32, bool)], value: T) -> Node<T> {
        let mut acc = self.make_leaf(value);
        if matches!(acc, Node::Neutral) {
            return acc;
        }

        for (input, index, value) in path.iter().rev().copied() {
            let mut arcs = [Node::Neutral, Node::Neutral];
            arcs[value as usize] = acc;
            acc = self.make_split(input, index, arcs);
        }

        acc
    }

    pub fn disjoint_union(&self, x: Node<T>, y: Node<T>) -> Result<Node<T>, &'static str> {
        match (x, y) {
            (Node::Neutral, other) | (other, Node::Neutral) => Ok(other),
            (Node::Leaf(x), Node::Leaf(y)) => {
                if x == y {
                    Ok(Node::Leaf(x))
                } else {
                    Err("May only union equal values")
                }
            }
            (Node::Split(split), leaf @ Node::Leaf(_))
            | (leaf @ Node::Leaf(_), Node::Split(split)) => Ok(self.make_split(
                split.input_id,
                split.index,
                [
                    self.disjoint_union(split.arcs[0].clone(), leaf.clone())?,
                    self.disjoint_union(split.arcs[1].clone(), leaf)?,
                ],
            )),
            (Node::Split(x), Node::Split(y)) => {
                if (x.input_id, x.index) == (y.input_id, y.index) {
                    Ok(self.make_split(
                        x.input_id,
                        x.index,
                        [
                            self.disjoint_union(x.arcs[0].clone(), y.arcs[0].clone())?,
                            self.disjoint_union(x.arcs[1].clone(), y.arcs[1].clone())?,
                        ],
                    ))
                } else {
                    Err("Can't union misaligned splits")
                }
            }
        }
    }

    pub fn merge(&self, x: Node<T>, y: Node<T>) -> Result<Node<T>, &'static str> {
        match (x, y) {
            (Node::Neutral, other) | (other, Node::Neutral) => Ok(other),
            (Node::Leaf(x), Node::Leaf(y)) => {
                let mut acc: T = (*x).clone();
                acc.merge((*y).clone());
                Ok(self.make_leaf(acc))
            }
            (leaf @ Node::Leaf(_), Node::Split(split))
            | (Node::Split(split), leaf @ Node::Leaf(_)) => Ok(self.make_split(
                split.input_id,
                split.index,
                [
                    self.merge(split.arcs[0].clone(), leaf.clone())?,
                    self.merge(split.arcs[1].clone(), leaf)?,
                ],
            )),
            (Node::Split(x), Node::Split(y)) => {
                if (x.input_id, x.index) == (y.input_id, y.index) {
                    Ok(self.make_split(
                        x.input_id,
                        x.index,
                        [
                            self.merge(x.arcs[0].clone(), y.arcs[0].clone())?,
                            self.merge(x.arcs[1].clone(), y.arcs[1].clone())?,
                        ],
                    ))
                } else {
                    Err("Can't merge misaligned splits")
                }
            }
        }
    }
}
