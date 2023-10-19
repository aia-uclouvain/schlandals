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

use super::graph::VariableIndex;
use search_trail::{StateManager, OptionBoolManager, ReversibleOptionBool};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Literal(isize, ReversibleOptionBool);

impl Literal {

    pub fn is_positive(&self) -> bool {
        self.0 > 0
    }
    
    pub fn to_variable(&self) -> VariableIndex {
        VariableIndex(self.0.abs() as usize - 1)
    }

    pub fn from_str(value: &str, fixed: ReversibleOptionBool) -> Self {
        Literal(value.parse::<isize>().unwrap(), fixed)
    }
    
    pub fn from_variable(variable: VariableIndex, polarity: bool, fixed: ReversibleOptionBool) -> Self {
        if polarity {
            Literal(variable.0 as isize + 1, fixed)
        } else {
            Literal(-(variable.0 as isize + 1), fixed)
        }
    }
    
    pub fn opposite(&self) -> Literal {
        Literal(-self.0, self.1)
    }
    
    pub fn is_variable_fixed(&self, state: &StateManager) -> bool {
        state.get_option_bool(self.1).is_some()
    }
}

impl std::fmt::Display for Literal {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        Ok(())
    }
}