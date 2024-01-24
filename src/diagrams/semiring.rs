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
use tch::Tensor;
use crate::common::f128;
use std::ops::{AddAssign, MulAssign};
use rug::Assign;

/// Trait that defines the behavior of a semiring.
pub trait SemiRing: AddAssign + MulAssign + Send + Sized + std::fmt::Display {

    /// Returns the neutral element for the multiplication
    fn one() -> Self {
        Self::from_f64(1.0)
    }
    /// Returns the neutral element for the addition
    fn zero() -> Self {
        Self::from_f64(0.0)
    }
    /// Returns the value, in the associated semiring, from the given f64
    fn from_f64(value: f64) -> Self;
    /// Returns the f64 value associated with the element of the semiring
    fn to_f64(&self) -> f64;
    /// Sets the value of the element to the given value
    fn set_value(&mut self, value: &Self);
    /// Copy the given element and returns the copy
    fn copy(from: &Self) -> Self;
    /// Create a new element on the semiring by summing all the elements given in the iterator
    fn sum_children<'a>(children: impl Iterator <Item = &'a Self>) -> Self
        where Self: 'a;
    /// Create a new element on the semiring by multiplying all the elements given in the iterator
    fn mul_children<'a>(children: impl Iterator <Item = &'a Self>) -> Self
        where Self: 'a;
}

impl SemiRing for Float {

    fn from_f64(value: f64) -> Self {
        f128!(value)
    }

    fn to_f64(&self) -> f64 {
        self.to_f64()
    }

    fn set_value(&mut self, value: &Float) {
        self.assign(value);
    }

    fn copy(from: &Self) -> Self {
        from.clone()
    }

    fn sum_children<'a>(children: impl Iterator <Item = &'a Self>) -> Self {
        let mut v = Self::zero();
        for child in children {
            v += child;
        }
        v
    }

    fn mul_children<'a>(children: impl Iterator <Item = &'a Self>) -> Self {
        let mut v = Self::one();
        for child in children {
            v *= child;
        }
        v
    }
}

impl SemiRing for Tensor {

    fn from_f64(value: f64) -> Self {
        Tensor::from(value)
    }

    fn to_f64(&self) -> f64 {
        self.f_double_value(&[]).unwrap()
    }

    fn set_value(&mut self, value: &Tensor) {
        self.copy_(value);
    }

    fn copy(other: &Self) -> Self {
        Tensor::copy(other)
    }

    fn sum_children<'a>(children: impl Iterator <Item = &'a Self>) -> Self {
        children.sum()
    }

    fn mul_children<'a>(children: impl Iterator <Item = &'a Self>) -> Self {
        let mut value = Self::one();
        for child in children {
            value *= child;
        }
        value
    }
}
