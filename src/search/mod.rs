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

pub mod sequential;
pub mod approximate;
mod statistics;

use crate::search::sequential::Solver as ExactSolver;
use crate::search::approximate::Solver as ApproximateSolver;

pub type ExactDefaultSolver<'b, B> = ExactSolver<'b, B, true>;
pub type ExactQuietSolver<'b, B> = ExactSolver<'b, B, false>;
pub type ApproximateDefaultSolver<'b, B> = ApproximateSolver<'b, B, true>;
pub type ApproximateQuietSolver<'b, B> = ApproximateSolver<'b, B, false>;

/// Unit structure representing the the problem is UNSAT
#[derive(Debug)]
pub struct Unsat;

/// Type alias used for the solution of the problem, which is either a Float or UNSAT
pub type ProblemSolution = Result<Float, Unsat>;