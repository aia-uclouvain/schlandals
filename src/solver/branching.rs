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

use crate::core::graph::DistributionIndex;

/// Trait that defined the methods that a branching decision structure must implement.
pub trait BranchingDecision {
    /// This function takes some distributions and returns an option with the next distribution to
    /// branch on (or None if `distributions.len() == 0`)
    fn branch_on(&mut self, distributions: &[DistributionIndex]) -> Option<DistributionIndex>;
}

/// A simple branching algorithm that selects the first distribution
#[derive(Default)]
pub struct FirstBranching;

impl BranchingDecision for FirstBranching {
    fn branch_on(&mut self, distributions: &[DistributionIndex]) -> Option<DistributionIndex> {
        if distributions.is_empty() {
            None
        } else {
            Some(distributions[0])
        }
    }
}

#[cfg(test)]
mod test_simple_branching {
    use super::{BranchingDecision, FirstBranching};
    use crate::core::graph::DistributionIndex;

    #[test]
    fn test_branching_distribution_order() {
        let mut distributions: Vec<DistributionIndex> =
            (0..5).map(|i| DistributionIndex(i)).collect();
        let mut b = FirstBranching::default();
        for i in 0..5 {
            assert_eq!(Some(DistributionIndex(i)), b.branch_on(&mut distributions));
            distributions.remove(0);
        }
        assert_eq!(None, b.branch_on(&mut distributions));
    }
}
