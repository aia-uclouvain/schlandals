//Schlandals
//Copyright (C) 2022 A. Dubray
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

use self::learner::Learner;
use self::approximate_learner::ApproximateLearner;

pub mod circuit;
pub mod learner;
pub mod approximate_learner;
pub mod exact;
mod logger;

pub type LogLearner = Learner<true>;
pub type QuietLearner = Learner<false>;

pub type LogApproximateLearner = ApproximateLearner<true>;
pub type QuietApproximateLearner = ApproximateLearner<false>;