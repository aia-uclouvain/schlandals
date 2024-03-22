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

use rug::Float;

macro_rules! f128 {
    ($v:expr) => {
        Float::with_val(113, $v)
    };
}

pub const FLOAT_CMP_THRESHOLD: f64 = 0.0000001;

pub struct Solution {
    probability: Float,
    epsilon: f64,
    sat: bool,
}

impl Solution {
    pub fn new(probability: Float, epsilon: f64, sat: bool) -> Self {
        Self { probability, epsilon, sat}
    }

    pub fn probability(&self) -> &Float { &self.probability }

    pub fn epsilon(&self) -> f64 { self.epsilon }

    pub fn is_sat(&self) -> bool { self.sat }
}

pub(crate) use f128;
