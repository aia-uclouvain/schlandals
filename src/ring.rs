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
    /// Apply the + operator between two elements of the ring and asigns it to the first argument
    fn plus(&self, a: &mut Float, b: &Float);
    /// App the x operator between two elements of the ring and assigns it to the first argument
    fn times(&self, a: &mut Float, b: &Float);
    /// Returns true if the ring is a counting one
    fn is_counting(&self)-> bool;
    fn is_log(&self) -> bool;
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
    fn plus(&self, a: &mut Float, b: &Float) {
        *a += b;
    }

    #[inline(always)]
    fn times(&self, a: &mut Float, b: &Float) {
        *a *= b;
    }

    #[inline(always)]
    fn is_counting(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_log(&self) -> bool {
        false
    }
}

#[derive(Copy, Clone, Default)]
pub struct LogAddMulRing;

impl Ring for LogAddMulRing {

    #[inline(always)]
    fn one(&self) -> Float {
        F128!(0.0)
    }

    #[inline(always)]
    fn zero(&self) -> Float {
        F128!(-f64::INFINITY)
    }

    #[inline(always)]
    fn plus(&self, a: &mut Float, b: &Float) {
        if *a > *b {
            *a += (b.clone() - a.clone()).exp10().log10_1p();
        } else {
            *a += (a.clone() - b.clone()).exp10().log10_1p();
        }
    }

    #[inline(always)]
    fn times(&self, a: &mut Float, b: &Float) {
        *a += b;
    }

    #[inline(always)]
    fn is_counting(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_log(&self) -> bool {
        true 
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
    fn plus(&self, a: &mut Float, b: &Float) {
        *a = a.clone().max(b);
    }

    #[inline(always)]
    fn times(&self, a: &mut Float, b: &Float) {
        *a *= b;
    }

    fn is_counting(&self) -> bool {
        false
    }

    fn is_log(&self) -> bool {
        true
    }
}
