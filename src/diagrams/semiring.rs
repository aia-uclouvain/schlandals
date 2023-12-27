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

use rug::Float;
use tch::Tensor;
use crate::common::f128;
use std::ops::{AddAssign, MulAssign};
use rug::Assign;

pub trait SemiRing: AddAssign + MulAssign + Send + Sized + std::fmt::Display {

    fn one() -> Self {
        Self::from_f64(1.0)
    }

    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    fn from_f64(value: f64) -> Self;
    fn to_f64(&self) -> f64;
    fn set_value(&mut self, value: &Self);
    fn copy(from: &Self) -> Self;
    fn sum_children<'a>(children: impl Iterator <Item = &'a Self>) -> Self
        where Self: 'a;
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
        Tensor::from_slice(&[value])
    }

    fn to_f64(&self) -> f64 {
        self.f_double_value(&[0]).unwrap()
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
