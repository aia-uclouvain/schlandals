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

pub trait SemiRing: AddAssign + MulAssign + Send + Sized {
    fn one() -> Self;
    fn zero() -> Self;
    fn from_float(value: f64) -> Self;
    fn to_f64(&self) -> f64;
    fn add_assign_ref(&mut self, other: &Self);
    fn mul_assign_ref(&mut self, other: &Self);
    fn backpropagating_gradient(&self) -> bool;
    fn gradient(&mut self, loss: f64);
}

impl SemiRing for Float {
    fn one() -> Self {
        f128!(1.0)
    }

    fn zero() -> Self {
        f128!(0.0)
    }

    fn from_float(value: f64) -> Self {
        f128!(value)
    }

    fn to_f64(&self) -> f64 {
        self.to_f64()
    }

    fn add_assign_ref(&mut self, other: &Self) {
        *self += other;
    }
    
    fn mul_assign_ref(&mut self, other: &Self) {
        *self *= other;
    }

    fn backpropagating_gradient(&self) -> bool {
        true
    }

    fn gradient(&mut self, _loss: f64) {
        
    }
}

impl SemiRing for Tensor {
    fn one() -> Self {
        Tensor::from_float(1.0)
    }

    fn zero() -> Self {
        Tensor::from_float(0.0)
    }

    fn from_float(value: f64) -> Self {
        Tensor::from_slice(&[value])
    }

    fn to_f64(&self) -> f64 {
        self.f_double_value(&[0]).unwrap()
    }

    fn add_assign_ref(&mut self, other: &Self) {
        *self += other;
    }

    fn mul_assign_ref(&mut self, other: &Self) {
        *self *= other;
    }

    fn backpropagating_gradient(&self) -> bool {
        false
    }

    fn gradient(&mut self, loss: f64) {
        self.backward();
    }
}
