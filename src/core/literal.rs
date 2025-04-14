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

    pub fn update_variable(&mut self, v: VariableIndex) {
        if self.is_positive() {
            self.0 = v.0 as isize + 1
        } else {
            self.0 = -(v.0 as isize + 1)
        }
    }
}

impl std::fmt::Display for Literal {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        Ok(())
    }
}
