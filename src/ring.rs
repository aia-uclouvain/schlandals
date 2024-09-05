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

//! This module provide definition and implementation of a semiring for various types.
//! A semiring is an algebraic structure over a set of elements S with an addition (+) and a
//! multiplication (*) such that
//!     1. There is an identity element for the + (0) (x + 0 = x and 0 + x = x)
//!     2. There is an identity element for the * (1) (x * 1 = x and 1 * x = x)
//!     3. The addition is commutative (x + y = y + x)
//!     4. The addition and multiplication is associative ( (a + b) + c = a + (b + c) and (a * b) * c = a * (b * c) )
use rug::Float;
use crate::common::F128;

/// Trait that defines the behavior of a semiring.
pub trait Ring: Sync {

    /// Returns the neutral element for the multiplication
    fn one(&self) -> Float;
    /// Returns the neutral element for the addition
    fn zero(&self) -> Float;
    /// Apply the + operator between two elements of the ring
    fn plus(&self, a: &Float, b: &Float) -> Float;
    /// App the x operator between two elements of the ring
    fn times(&self, a: &Float, b: &Float) -> Float;
}

#[derive(Copy, Clone, Default)]
pub struct AddMulRing;

impl Ring for AddMulRing {

    #[inline(always)]
    fn one(&self) -> Float {
        F128!(1.0)
    }

    #[inline(always)]
    fn zero(&self) -> Float {
        F128!(0.0)
    }

    #[inline(always)]
    fn plus(&self, a: &Float, b: &Float) -> Float {
        a.clone() + b
    }

    #[inline(always)]
    fn times(&self, a: &Float, b: &Float) -> Float {
        a.clone() * b
    }
}

#[derive(Copy, Clone, Default)]
pub struct MaxMulRing;

impl Ring for MaxMulRing {

    #[inline(always)]
    fn one(&self) -> Float {
        F128!(1.0)
    }

    #[inline(always)]
    fn zero(&self) -> Float {
        F128!(0.0)
    }

    #[inline(always)]
    fn plus(&self, a: &Float, b: &Float) -> Float {
        a.clone().max(b)
    }

    #[inline(always)]
    fn times(&self, a: &Float, b: &Float) -> Float {
        a.clone() * b
    }
}
