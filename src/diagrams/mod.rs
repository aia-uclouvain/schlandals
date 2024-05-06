//Schlandals
//Copyright (C) 2022 A. Dubray, L. Dierckx
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

pub mod dac;
pub mod semiring;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Types of node in an AC
pub enum TypeNode {
    /// Product nodes
    Product,
    /// Sum nodes
    Sum,
    /// Approximate node. Only present when the circuit is partially compiled. Send a constant
    /// value as output and act as input of the circuit
    Approximate,
    /// Distribution node. Send the value P[d = v] as output and act as input of the circuit
    Distribution {d: usize, v: usize},
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeStatus {
    Sat,
    Unsat,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(pub usize);

macro_rules! is_node_type {
    ($val:expr, $var:path) => {
        match $val {
            $var{..} => true,
            _ => false,
        }
    }
}

use is_node_type;
