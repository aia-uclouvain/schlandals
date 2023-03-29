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

use std::fmt;

/// Implements a bunch of statistics that are collected during the search
#[derive(Default)]
#[cfg(not(tarpaulin_include))]
pub struct Statistics<const B: bool> {
    cache_miss: usize,
    cache_access: usize,
    number_or_nodes: usize,
    number_and_nodes: usize,
    total_and_decompositions: usize,
}

impl<const B: bool> Statistics<B> {
    pub fn cache_miss(&mut self) {
        if B {
            self.cache_miss += 1;
        }
    }

    pub fn cache_access(&mut self) {
        if B {
            self.cache_access += 1;
        }
    }

    pub fn or_node(&mut self) {
        if B {
            self.number_or_nodes += 1;
        }
    }

    pub fn and_node(&mut self) {
        if B {
            self.number_and_nodes += 1;
        }
    }

    pub fn decomposition(&mut self, number_components: usize) {
        if B {
            self.total_and_decompositions += number_components;
        }
    }

    pub fn print(&self) {
        if B {
            println!("{}", self);
        }
    }
}

impl<const B: bool> fmt::Display for Statistics<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if B {
            writeln!(f, "Statistics on the search:")?;
            writeln!(f, "\tNumber of cache access: {}", self.cache_access)?;
            writeln!(
                f,
                "\tNumber of cache hit: {} ({:.3} %)",
                self.cache_access - self.cache_miss,
                100f64 - (self.cache_miss as f64 / self.cache_access as f64) * 100.0
            )?;
            writeln!(f, "\tNumber of OR nodes: {}", self.number_or_nodes)?;
            writeln!(f, "\tNumber of AND nodes: {}", self.number_and_nodes)?;
            writeln!(
                f,
                "\tAverage sub-problem decomposition per AND node: {:.3}",
                (self.total_and_decompositions as f64) / (self.number_and_nodes as f64)
            )
        } else {
            write!(f, "")
        }
    }
}
