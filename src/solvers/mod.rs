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


use rug::Float;
/// Unit structure representing the the problem is UNSAT
#[derive(Debug)]
pub struct Unsat;

/// Type alias used for the solution of the problem, which is either a Float or UNSAT
pub type ProblemSolution = Result<Float, Unsat>;

pub type Bounds = (Float, Float);

pub mod search;
pub mod compiler;
pub mod lds;
pub mod sampler;
mod statistics;

use search::SearchSolver;
use lds::LDSSolver;
pub use sampler::SamplerSolver;

pub type StatSearchSolver<'b, B> = SearchSolver<'b, B, true>;
pub type QuietSearchSolver<'b, B> = SearchSolver<'b, B, false>;
pub type StatLDSSolver<'b, B> = LDSSolver<'b, B, true>;
pub type QuietLDSSolver<'b, B> = LDSSolver<'b, B, false>;
