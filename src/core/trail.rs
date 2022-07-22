//Schlandals
//Copyright (C) 2022 A. Dubray
//
//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU Affero General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU Affero General Public License for more details.
//
//You should have received a copy of the GNU Affero General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//! This module implements a trailing mechanism that is able to save and restore states for
//! integers, booleans and floats (which are, at the moment, the only values we need to manage for
//! this project)
//! This code is taken from [maxi-cp-rs](https://github.com/xgillard/maxicp-rs) (only the parts
//! that interest us)

/// The identifier of a managed integer resource
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReversibleInt(usize);

/// The indentifier of a managed boolean resource (which is mapped to a reversible integer)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReversibleBool(ReversibleInt);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReversibleFloat(usize);

pub trait StateManager: SaveAndRestore + IntManager + BoolManager + FloatManager {}

pub trait SaveAndRestore {
    /// Saves the current state of all managed resources
    fn save_state(&mut self);

    /// Restores the previous state of all managed resources
    fn restore_state(&mut self);
}

/// All the operations that can be done on a managed integer resource
pub trait IntManager {
    /// Creates a new managed integer
    fn manage_int(&mut self, value: isize) -> ReversibleInt;
    /// Returns the value of a managed integer
    fn get_int(&self, id: ReversibleInt) -> isize;
    /// Sets the value of a managed integer
    fn set_int(&mut self, id: ReversibleInt, value: isize) -> isize;
    /// Increments the value of a managed integer
    fn increment(&mut self, id: ReversibleInt) -> isize;
    /// Decrements the value of a managed integer
    fn decrement(&mut self, id: ReversibleInt) -> isize;
}

/// All the operations that can be done on a managed boolean resource
pub trait BoolManager {
    /// Creates a new managed boolean
    fn manage_boolean(&mut self, value: bool) -> ReversibleBool;
    /// Returns the value of a managed boolean
    fn get_bool(&self, id: ReversibleBool) -> bool;
    /// Sets the value of a managed boolean
    fn set_bool(&mut self, id: ReversibleBool, value: bool) -> bool;
    /// Flips the value of a managed boolean
    fn flip_bool(&mut self, id: ReversibleBool) -> bool {
        self.set_bool(id, !self.get_bool(id))
    }
}

/// All the operations that can be done on a managed float resource
pub trait FloatManager {
    /// Creates a new managed float
    fn manage_float(&mut self, value: f64) -> ReversibleFloat;
    /// Returns the value of a  managed float
    fn get_float(&self, id: ReversibleFloat) -> f64;
    /// Sets the value of a managed float
    fn set_float(&mut self, id: ReversibleFloat, value: f64) -> f64;
    /// Adds `value` to the managed float
    fn add_float(&mut self, id: ReversibleFloat, value: f64) -> f64;
    /// Substracts `value` of the managed float
    fn substract_float(&mut self, id: ReversibleFloat, value: f64) -> f64;
}

/// This structure keeps track of the length of a given level of the trail as well as the number of
/// managed resources of each kind. This second information is useful in order to truncate the
/// vector in the state manager.
#[derive(Debug, Clone, Copy, Default)]
struct Level {
    /// The length of the trail at the moment this level was started
    trail_size: usize,
    /// How many integers were recorded at the moment this level was started
    integers: usize,
    /// How many floats were recorded at the moment this level was started
    floats: usize,
}

/// An entry that is used to restore data from the trail
#[derive(Debug, Clone, Copy)]
enum TrailEntry {
    /// An entry related to the restoration of an integer
    IntEntry(IntState),
    /// An entry related to the restoration of a float
    FloatEntry(FloatState),
}

#[derive(Debug, Clone)]
pub struct TrailedStateManager {
    /// This clock is responsible to tell if a data need to be stored on the trail for restitution
    /// or not. If a managed resource X is changed and X.clock < clock, then it needs to be saved
    /// on the trail for restitution. Once the managed resource is updated, X.clock = clock.
    ///
    /// This clock is incremented at each call to `save_state()`
    clock: usize,
    /// The values that are saved on the trail. These entries are used to restore the managed
    /// resources when `restore_state()` is called
    trail: Vec<TrailEntry>,
    /// Levels of the trail where a level is an indicator of the number of `TrailEntry` for a given
    /// timestamp of `clock`
    levels: Vec<Level>,
    /// The current values of the managed integers and booleans (mapped to an int)
    integers: Vec<IntState>,
    /// The current values of the managed floats
    floats: Vec<FloatState>,
}

impl Default for TrailedStateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TrailedStateManager {
    pub fn new() -> Self {
        Self {
            clock: 0,
            trail: vec![],
            levels: vec![Level {
                trail_size: 0,
                integers: 0,
                floats: 0,
            }],
            integers: vec![],
            floats: vec![],
        }
    }
}

impl StateManager for TrailedStateManager {}

// --- Save and restore --- //

impl SaveAndRestore for TrailedStateManager {
    fn save_state(&mut self) {
        // Increment the clock of the state manager. After this, every managed resource will become
        // "invalid" and will need to be stored on the trail if changed
        self.clock += 1;
        self.levels.push(Level {
            trail_size: self.trail.len(),
            integers: self.integers.len(),
            floats: self.floats.len(),
        });
    }

    fn restore_state(&mut self) {
        debug_assert!(self.levels.len() > 1);
        let level = self
            .levels
            .pop()
            .expect("Can not pop the root level of the state manager");

        // Before the creation of the current level, the trail was `trail_size` long, so we skip
        // these first elements.
        for e in self.trail.iter().skip(level.trail_size).rev().copied() {
            match e {
                TrailEntry::IntEntry(state) => self.integers[state.id.0] = state,
                TrailEntry::FloatEntry(state) => self.floats[state.id.0] = state,
            }
        }

        self.trail.truncate(level.trail_size);
        self.integers.truncate(level.integers);
        self.floats.truncate(level.floats);
    }
}

// --- Int management --- //

/// State of an integer that can be saved and restored
#[derive(Debug, Clone, Copy)]
struct IntState {
    /// The id of the managed integer
    id: ReversibleInt,
    /// Clock of the data. If less than the clock of the manager, this data needs to be saved on
    /// the trail when modified
    clock: usize,
    /// The actual value of the integer
    value: isize,
}

impl IntManager for TrailedStateManager {
    fn manage_int(&mut self, value: isize) -> ReversibleInt {
        let id = ReversibleInt(self.integers.len());
        self.integers.push(IntState {
            id,
            clock: self.clock,
            value,
        });
        id
    }

    fn get_int(&self, id: ReversibleInt) -> isize {
        self.integers[id.0].value
    }

    fn set_int(&mut self, id: ReversibleInt, value: isize) -> isize {
        let curr = self.integers[id.0];
        // If the value if the same as already present in the state, we do not need to do anything
        if value != curr.value {
            // Two cases:
            //  - The clock is less than the manager clock. Then the data needs to be saved on the
            //  trail for restoring.
            //  - The clock is equal to the  manager clock. Then there were no `save_state()` call
            //  since the last modification (or creation) and the data can be modified directly
            if curr.clock < self.clock {
                self.trail.push(TrailEntry::IntEntry(curr));
                self.integers[id.0] = IntState {
                    id,
                    clock: self.clock,
                    value,
                };
            } else {
                self.integers[id.0].value = value;
            }
        }
        value
    }

    fn increment(&mut self, id: ReversibleInt) -> isize {
        self.set_int(id, self.get_int(id) + 1)
    }

    fn decrement(&mut self, id: ReversibleInt) -> isize {
        self.set_int(id, self.get_int(id) - 1)
    }
}

// --- Bool management --- //

impl BoolManager for TrailedStateManager {
    fn manage_boolean(&mut self, value: bool) -> ReversibleBool {
        ReversibleBool(self.manage_int(value as isize))
    }

    fn get_bool(&self, id: ReversibleBool) -> bool {
        self.get_int(id.0) != 0
    }

    fn set_bool(&mut self, id: ReversibleBool, value: bool) -> bool {
        self.set_int(id.0, value as isize) != 0
    }
}

// --- Float management --- //

/// State of a float that can be saved and restore
#[derive(Debug, Clone, Copy)]
struct FloatState {
    /// The id of the managed float
    id: ReversibleFloat,
    /// Clock of the data. If less than the clock of the manager, this data needs to be saved on
    /// the trail when modified
    clock: usize,
    /// The actual value of the float
    value: f64,
}

impl FloatManager for TrailedStateManager {
    fn manage_float(&mut self, value: f64) -> ReversibleFloat {
        let id = ReversibleFloat(self.floats.len());
        self.floats.push(FloatState {
            id,
            clock: self.clock,
            value,
        });
        id
    }

    fn get_float(&self, id: ReversibleFloat) -> f64 {
        self.floats[id.0].value
    }

    fn set_float(&mut self, id: ReversibleFloat, value: f64) -> f64 {
        let curr = self.floats[id.0];
        if value != curr.value {
            if curr.clock < self.clock {
                self.trail.push(TrailEntry::FloatEntry(curr));
                self.floats[id.0] = FloatState {
                    id,
                    clock: self.clock,
                    value,
                };
            } else {
                self.floats[id.0].value = value;
            }
        }
        value
    }

    fn add_float(&mut self, id: ReversibleFloat, value: f64) -> f64 {
        let v = self.get_float(id) + value;
        self.set_float(id, v)
    }

    fn substract_float(&mut self, id: ReversibleFloat, value: f64) -> f64 {
        let v = self.get_float(id) - value;
        self.set_float(id, v)
    }
}

#[cfg(test)]
mod test_manager {
    use crate::core::trail::*;

    #[test]
    #[should_panic]
    fn can_not_get_bool_manage_at_deeper_level() {
        let mut mgr = TrailedStateManager::new();
        let a = mgr.manage_boolean(true);
        assert!(mgr.get_bool(a));

        mgr.save_state();

        let b = mgr.manage_boolean(false);
        assert!(!mgr.get_bool(b));
        assert!(mgr.get_bool(a));

        mgr.set_bool(a, false);

        mgr.restore_state();
        assert!(mgr.get_bool(a));
        mgr.get_bool(b);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn can_not_pop_root_level() {
        let mut mgr = TrailedStateManager::new();
        let a = mgr.manage_boolean(true);

        mgr.save_state();
        mgr.set_bool(a, false);
        mgr.restore_state();
        mgr.restore_state();
    }
}

#[cfg(test)]
mod test_manager_bool {

    use crate::core::trail::*;

    #[test]
    fn works() {
        let mut mgr = TrailedStateManager::new();
        let a = mgr.manage_boolean(false);
        assert!(!mgr.get_bool(a));

        mgr.save_state();

        let x = mgr.set_bool(a, true);
        assert!(x);
        assert!(mgr.get_bool(a));

        mgr.restore_state();
        assert!(!mgr.get_bool(a));

        let x = mgr.flip_bool(a);
        assert!(x);
        mgr.save_state();

        let x = mgr.set_bool(a, false);
        assert!(!x);
        let x = mgr.set_bool(a, true);
        assert!(x);
        assert!(mgr.get_bool(a));
        mgr.restore_state();
        assert!(mgr.get_bool(a));
    }
}

#[cfg(test)]
mod test_manager_integer {

    use crate::core::trail::*;

    #[test]
    fn set_and_restore_works() {
        let mut mgr = TrailedStateManager::new();
        let a = mgr.manage_int(10);
        assert_eq!(10, mgr.get_int(a));

        mgr.save_state();

        let x = mgr.set_int(a, 20);
        assert_eq!(20, x);
        assert_eq!(20, mgr.get_int(a));

        let x = mgr.set_int(a, 23);
        assert_eq!(23, x);
        assert_eq!(23, mgr.get_int(a));

        mgr.restore_state();
        assert_eq!(10, mgr.get_int(a));

        let x = mgr.set_int(a, 42);
        assert_eq!(42, x);
        assert_eq!(42, mgr.get_int(a));

        mgr.save_state();

        let x = mgr.set_int(a, 12);
        assert_eq!(12, x);
        assert_eq!(12, mgr.get_int(a));

        mgr.save_state();

        let x = mgr.set_int(a, 12);
        assert_eq!(12, x);
        assert_eq!(12, mgr.get_int(a));
        mgr.save_state();

        mgr.restore_state();

        assert_eq!(12, mgr.get_int(a));

        mgr.restore_state();
        assert_eq!(12, mgr.get_int(a));

        mgr.restore_state();
        assert_eq!(42, mgr.get_int(a));
    }

    #[test]
    fn test_increments() {
        let mut mgr = TrailedStateManager::new();
        let a = mgr.manage_int(0);
        assert_eq!(0, mgr.get_int(a));

        mgr.save_state();

        for i in 0..10 {
            let x = mgr.increment(a);
            assert_eq!(i + 1, x);
            assert_eq!(i + 1, mgr.get_int(a));
        }

        mgr.restore_state();
        assert_eq!(0, mgr.get_int(a));

        mgr.save_state();

        for i in 0..10 {
            let x = mgr.decrement(a);
            assert_eq!(-i - 1, x);
            assert_eq!(-i - 1, mgr.get_int(a));
        }

        mgr.restore_state();
        assert_eq!(0, mgr.get_int(a));
    }
}

#[cfg(test)]
mod test_manager_float {
    use crate::core::trail::*;

    #[test]
    fn set_and_restore_works() {
        let mut mgr = TrailedStateManager::new();
        let a = mgr.manage_float(0.3);
        assert_eq!(0.3, mgr.get_float(a));

        mgr.save_state();
        let x = mgr.set_float(a, 0.5);
        assert_eq!(0.5, x);
        assert_eq!(0.5, mgr.get_float(a));

        mgr.save_state();
        let x = mgr.set_float(a, 1.5);
        assert_eq!(1.5, x);
        assert_eq!(1.5, mgr.get_float(a));

        mgr.save_state();
        let x = mgr.set_float(a, -12.3);
        assert_eq!(-12.3, x);
        assert_eq!(-12.3, mgr.get_float(a));

        mgr.restore_state();
        assert_eq!(1.5, mgr.get_float(a));

        mgr.restore_state();
        assert_eq!(0.5, mgr.get_float(a));

        mgr.restore_state();
        assert_eq!(0.3, mgr.get_float(a));
    }

    #[test]
    fn add_and_substract_floats() {
        let mut mgr = TrailedStateManager::new();
        let f = mgr.manage_float(0.0);
        assert_eq!(0.0, mgr.get_float(f));

        mgr.add_float(f, 1.5);
        assert_eq!(1.5, mgr.get_float(f));

        mgr.save_state();

        mgr.substract_float(f, 12.2);
        assert_eq!(-10.7, mgr.get_float(f));
        mgr.add_float(f, 1.0);
        assert_eq!(-9.7, mgr.get_float(f));
        mgr.restore_state();
        assert_eq!(1.5, mgr.get_float(f));
    }
}
