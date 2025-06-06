//! This module implements a reversible sparse-set for efficient domain iterations.
//! A sparse-set is a data structure composed of three elements:
//!     1. A plain vector, containing the elements of the set
//!     2. A map of index, mapping the elements to their index in the plain vector
//!     3. The current size of the sparse set
//! 
//! For example, the representation of the set {30, 5, 12, 45} might be
//!     1. [30, 5, 12, 45]
//!     2. [30 -> 0, 5 -> 1, 12 -> 2, 45 -> 3]
//!     3. 4
//! 
//! To remove an element from the set, it can be swapped at the end of the plain vector and
//! the size decremented. For example, if we remove 5 from the above set, we have the following
//! representation (using | as a delimiter for the size)
//!     1. [30, 45, 12, | 5]
//!     2. [30 -> 0, 5 -> 3, 12 -> 2, 45 -> 1]
//!     3. 3
//! 
//! It can be seen that by doing so, the removal is in O(1) and iterating over the remaining elements
//! can be done in efficiently (not visiting removed elements).
//! To make it reversible, the size is stored as a RervsibleUsize from the search_trail crate.
//! If we have the set {30, 5, 12, 45} and we do the following operations (in that order):
//!     - state.save_state();
//!     - set.remove(5);
//!     - state.restore_state();
//! 
//! Then the reversible sparse set is in the following state
//!     1. [30, 45, 12, 5]
//!     2. [30 -> 0, 5 -> 3, 12 -> 2, 45 -> 1]
//!     3. 4
//!     
//! In short, to reverse the set to its previous state, only reverting the size of the set is needed.
//! Hence this is a O(1) operation.

use rustc_hash::FxHashMap;
use std::hash::Hash;
use search_trail::{StateManager, ReversibleUsize, UsizeManager};

#[derive(Debug)]
/// Structure that implements a generic reversible sparse-set over elements of type T.
/// We require that the type T can be used as key in a hashmap for storing the indexes.
/// For our purpose, we suppose that the set is used in the following manner:
///     1. First, all the elements are added in the set.
///     2. Then the set is used in the search and is updated/restored
/// 
/// We assume that **no elements are added in the set after the first call to remove**
pub struct SparseSet<T> 
    where T: Hash + Eq + Copy,
{
    /// Vectors containing the elements in the set
    plain: Vec<T>,
    /// Map each elements of the set to its index in the plain vector
    indexes: FxHashMap<T, usize>,
    /// Size of the set
    removed: ReversibleUsize,
}

#[allow(clippy::len_without_is_empty)]
impl<T> SparseSet<T> 
    where T: Hash + Eq + Copy,
{
    
    /// Creates a new empty reversible sparse-set
    pub fn new(state: &mut StateManager) -> Self {
        Self {
            plain: vec![],
            indexes: FxHashMap::<T, usize>::default(),
            removed: state.manage_usize(0),
        }
    }
    
    /// Adds element `eleme` to the sparse-set
    pub fn add(&mut self, elem: T, state: &mut StateManager) {
        if self.indexes.contains_key(&elem) {
            return;
        }
        // Note: this is a strange way to insert into a sparse-set. But with clause learning, it
        // happens that a clause is added during the search, hence while the sparse-set is not
        // full. This ensure that the element is added in the "alive" part of the set, and that the
        // indexes are update
        let index = self.plain.len() - state.get_usize(self.removed);
        self.indexes.insert(elem, index);
        self.plain.insert(index, elem);
        for i in (index + 1)..self.plain.len() {
            self.indexes.insert(self.plain[i], i);
        }
    }
    
    /// Removes element `elem` from the sparse-set
    pub fn remove(&mut self, elem: T, state: &mut StateManager) {
        let cur_idx = *self.indexes.get(&elem).unwrap();
        if cur_idx >= self.len(state) {
            return;
        }
        let last_idx = self.plain.len() - 1 - state.get_usize(self.removed);
        self.plain.swap(cur_idx, last_idx);
        self.indexes.insert(self.plain[cur_idx], cur_idx);
        self.indexes.insert(self.plain[last_idx], last_idx);
        state.increment_usize(self.removed);
    }

    pub fn remove_all(&self, state: &mut StateManager) {
        state.set_usize(self.removed, self.plain.len());
    }
    
    /// Iterates over the current elements of the sparse-set
    pub fn iter(&self, state: &StateManager) -> impl Iterator<Item = T> + '_ {
        self.plain[0..self.len(state)].iter().copied()
    }
    
    /// Returns the current size of the sparse-set
    pub fn len(&self, state: &StateManager) -> usize {
        self.plain.len() - state.get_usize(self.removed)
    }
    
    /// Returns the total capacity of the sparse-set (i.e., the total number of element in
    /// the set, including the removed ones)
    pub fn capacity(&self) -> usize {
        self.plain.len()
    }

    pub fn clear(&mut self, map: &FxHashMap<T, T>, state: &mut StateManager) {
        state.set_usize(self.removed, 0);
        for i in (0..self.plain.len()).rev() {
            let elem = self.plain[i];
            match map.get(&elem).copied() {
                Some(new_elem) => { self.plain[i] = new_elem; },
                None => { self.plain.swap_remove(i); },
            };
        }
        self.indexes.clear();
        for (i, elem) in self.plain.iter().copied().enumerate() {
            self.indexes.insert(elem, i);
        }
        self.plain.shrink_to_fit();
    }
}

#[cfg(test)]
mod test_sparse_set {
    use search_trail::{StateManager, SaveAndRestore};
    use crate::core::sparse_set::*;
    
    fn check_map(set: &SparseSet<usize>, expected: Vec<(usize, usize)>) {
        assert_eq!(expected.len(), set.indexes.len());
        for (k, v) in expected {
            let in_map = set.indexes.get(&k);
            assert!(in_map.is_some());
            assert_eq!(v, *in_map.unwrap());
        }
    }
    
    #[test]
    pub fn initialization() {
        let mut state = StateManager::default();
        let set = SparseSet::<usize>::new(&mut state);
        assert!(set.plain.is_empty());
        assert_eq!(0, set.len(&state));
        check_map(&set, vec![]);
    }
    
    #[test]
    pub fn add_elments() {
        let mut state = StateManager::default();
        let mut set = SparseSet::<usize>::new(&mut state);
        set.add(10, &mut state);
        set.add(5, &mut state);
        set.add(43, &mut state);
        set.add(55, &mut state);
        assert_eq!(4, set.len(&state));
        assert_eq!(vec![10, 5, 43, 55], set.plain);
        check_map(&set, vec![(10, 0), (5, 1), (43, 2), (55, 3)]);
    }
    
    #[test]
    pub fn remove_elements() {
        let mut state = StateManager::default();
        let mut set = SparseSet::<usize>::new(&mut state);
        set.add(10, &mut state);
        set.add(5, &mut state);
        set.add(43, &mut state);
        set.add(55, &mut state);
        
        set.remove(5, &mut state);
        assert_eq!(3, set.len(&state));
        assert_eq!(vec![10, 55, 43, 5], set.plain);
        check_map(&set, vec![(10, 0), (55, 1), (43, 2), (5, 3)]);
    }
    
    #[test]
    pub fn restore_state() {
        let mut state = StateManager::default();
        let mut set = SparseSet::<usize>::new(&mut state);
        set.add(10, &mut state);
        set.add(5, &mut state);
        set.add(43, &mut state);
        set.add(55, &mut state);
        
        state.save_state();
        
        set.remove(5, &mut state);
        
        state.restore_state();
        assert_eq!(4, set.len(&state));
        assert_eq!(vec![10, 55, 43, 5], set.plain);
        check_map(&set, vec![(10, 0), (55, 1), (43, 2), (5, 3)]);
    }
}
