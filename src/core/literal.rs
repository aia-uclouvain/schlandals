//Schlandals
//Copyright (C) 2022-2023 A. Dubray
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

//! An implementation of a literal in Schlandals. That is, a variable an a
//! polarity. This is represented by a signed integer.
//! In addition to the variable and its polarity (the isize), each literal has the
//! reversible boolean representing the fact that the associated variable is fixed or
//! not.
//! This is done so that each literal can query on its own the state of its associated variable
//! (bypassing problems with the borrow checker).

use super::problem::VariableIndex;
use search_trail::{StateManager, OptionBoolManager, ReversibleOptionBool};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Literal(isize, ReversibleOptionBool);

impl Literal {

    /// Returns true iff the literal has a positive polarity
    pub fn is_positive(&self) -> bool {
        self.0 > 0
    }
    
    /// Returns the varaible represented by the literal
    pub fn to_variable(&self) -> VariableIndex {
        VariableIndex(self.0.unsigned_abs() - 1)
    }

    /// Returns a literal from its string representation
    pub fn from_str(value: &str, fixed: ReversibleOptionBool) -> Self {
        Literal(value.parse::<isize>().unwrap(), fixed)
    }
    
    /// Returns the literal representing the variable with the given polarity
    pub fn from_variable(variable: VariableIndex, polarity: bool, fixed: ReversibleOptionBool) -> Self {
        if polarity {
            Literal(variable.0 as isize + 1, fixed)
        } else {
            Literal(-(variable.0 as isize + 1), fixed)
        }
    }
    
    /// Returns the opposite of the current literal. That is, a literal representing the same
    /// variable but with opposite polarity
    pub fn opposite(&self) -> Literal {
        Literal(-self.0, self.1)
    }
    
    /// Returns true iff the associated variable is fixed
    pub fn is_variable_fixed(&self, state: &StateManager) -> bool {
        state.get_option_bool(self.1).is_some()
    }

    pub fn trail_index(&self) -> ReversibleOptionBool {
        self.1
    }
}

impl std::fmt::Display for Literal {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        Ok(())
    }
}
