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

#[derive(Clone, Debug, Default)]
pub struct Bitvec {
    vec: Vec<u128>,
}

pub const WORD_SIZE: usize = 128;

impl Bitvec {

    pub fn new(number_element: usize) -> Self {
        let number_word = (number_element as f64 / 64_f64).ceil() as usize;
        Self {
            vec: vec![0_u128; number_word],
        }
    }

    pub fn ones(number_element: usize) -> Self {
        let number_word = (number_element as f64 / 64_f64).ceil() as usize;
        Self {
            vec: vec![!0_u128; number_word],
        }
    }

    #[inline(always)]
    pub fn set_bit(&mut self, word_index: usize, mask: u128) {
        self.vec[word_index] |= mask;
    }
}

impl PartialEq for Bitvec {
    fn eq(&self, other: &Self) -> bool {
        self.vec == other.vec
    }
}

impl Eq for Bitvec {}
