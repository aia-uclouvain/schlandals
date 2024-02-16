//Schlandal
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
use clap::ValueEnum;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum DiscrepancyStrategy {
    /// Increment the discrepancy by a constant factor
    Monotonic,
    /// Increment the discrepancy exponentially for the given base
    Exponential,
    /// Increment the discrepancy following the luby procedure, with a constant multiplier
    Luby,
}

impl DiscrepancyStrategy {

    pub fn to_strategy(&self, arg: usize) -> Box<dyn Discrepancy> {
        match self {
            DiscrepancyStrategy::Monotonic => Box::new(MonotonicDiscrepancy::new(arg)),
            DiscrepancyStrategy::Exponential => Box::new(ExponentialDiscrepancy::new(arg)),
            DiscrepancyStrategy::Luby => Box::new(LubyDiscrepancy::new(arg)),
        }
    }
}

pub trait Discrepancy {
    fn update_discrepancy(&mut self);
    fn discrepancy(&self) -> usize;
}

pub struct MonotonicDiscrepancy {
    discrepancy: usize,
    increment: usize,
}

impl MonotonicDiscrepancy {

    pub fn new(increment: usize) -> Self {
        Self {
            discrepancy: increment,
            increment
        }
    }
}

impl Discrepancy for MonotonicDiscrepancy {

    fn update_discrepancy(&mut self) {
        self.discrepancy += self.increment;
    }

    fn discrepancy(&self) -> usize {
        self.discrepancy
    }
}

pub struct ExponentialDiscrepancy {
    discrepancy: usize,
    base: usize,
}

impl ExponentialDiscrepancy {
    
    pub fn new(base: usize) -> Self {
        Self {
            discrepancy: 1,
            base
        }
    }
}

impl Discrepancy for ExponentialDiscrepancy {

    fn update_discrepancy(&mut self) {
        self.discrepancy *= self.base;
    }

    fn discrepancy(&self) -> usize {
        self.discrepancy
    }
}

pub struct LubyDiscrepancy {
    discrepancy: usize,
    iter: usize,
    increments: Vec<usize>,
    multiplier: usize,
}

impl LubyDiscrepancy {

    pub fn new(multiplier: usize) -> Self {
        Self {
            discrepancy: 1,
            iter: 1,
            increments: vec![],
            multiplier,
        }
    }
}

impl Discrepancy for LubyDiscrepancy {

    fn update_discrepancy(&mut self) {
        if (self.iter + 1) % 2 == 0 {
            self.discrepancy = 2_usize.pow((self.iter + 1).ilog2()) * self.multiplier;
        } else {
            let index = self.iter - 2_usize.pow((self.iter as u32 / 2 ) - 1);
            self.discrepancy = self.increments[index];
        }
        self.increments.push(self.discrepancy);
    }

    fn discrepancy(&self) -> usize {
        self.discrepancy
    }
}

impl std::fmt::Display for DiscrepancyStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiscrepancyStrategy::Monotonic => write!(f, "mono"),
            DiscrepancyStrategy::Exponential => write!(f, "exp"),
            DiscrepancyStrategy::Luby => write!(f, "luby"),
        }
    }
}
