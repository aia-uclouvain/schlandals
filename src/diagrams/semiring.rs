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
use crate::Loss;
use tch::Reduction;

pub trait SemiRing: AddAssign + MulAssign + Send + Sized + std::fmt::Display {

    fn one() -> Self {
        Self::from_f64(1.0, false)
    }

    fn zero() -> Self {
        Self::from_f64(0.0, false)
    }

    fn from_f64(value: f64, require_grad: bool) -> Self;
    fn to_f64(&self) -> f64;
    fn add_assign_ref(&mut self, other: &Self);
    fn mul_assign_ref(&mut self, other: &Self);
    fn backpropagating_gradient(&self) -> bool;
    fn do_backward(&mut self, loss: Loss, target: f64);
    fn gradient(&self) -> f64;
}

impl SemiRing for Float {

    fn from_f64(value: f64, _require_grad: bool) -> Self {
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

    fn do_backward(&mut self, _loss: Loss, _target: f64) {
        
    }

    fn gradient(&self) -> f64 {
        0.0
    }
}

impl SemiRing for Tensor {

    fn from_f64(value: f64, require_grad: bool) -> Self {
        let t = Tensor::from_slice(&[value]);
        t.set_requires_grad(require_grad)
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

    fn do_backward(&mut self, loss: Loss, target: f64) {
        let y = Self::from_f64(target, false);
        match loss {
            Loss::MAE => {
                let loss = self.f_l1_loss(&y, Reduction::Mean).unwrap();
                loss.backward();
            },
            Loss::MSE => {
                let loss = self.mse_loss(&y, Reduction::Mean);
                loss.backward();
            },
        };
    }

    fn gradient(&self) -> f64 {
        self.f_grad().unwrap().f_double_value(&[0]).unwrap()
    }
}
