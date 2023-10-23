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

//  Inspired from @xgillard rsolve (https://www.github.com/xgillard/rsolve)
pub enum ClauseFlag {
    Clear = 0,
    TrueReachable = 1,
    FalseReachable = 2,    
}

#[derive(Clone, Copy)]
pub struct ClauseFlags(u8);

impl ClauseFlags {

    #[inline]
    pub fn new() -> Self {
        ClauseFlags(ClauseFlag::Clear as u8)
    }
    
    #[inline]
    pub fn set(&mut self, flag: ClauseFlag) {
        self.0 |= flag as u8;
    }
    
    #[inline]
    pub fn unset(&mut self, flag: ClauseFlag) {
        self.0 ^= flag as u8;
    }
    
    #[inline]
    pub fn clear(&mut self) {
        self.0 = ClauseFlag::Clear as u8;
    }
    
    #[inline]
    pub fn is_set(&self, flag: ClauseFlag) -> bool {
        self.0 & (flag as u8) != 0
    }
    
    #[inline]
    pub fn is_reachable(&self) -> bool {
        self.0 ^ (ClauseFlag::FalseReachable as u8 | ClauseFlag::TrueReachable as u8) == 0
    } 
}

pub enum LitFlag {
    Clear = 0,
    IsMarked = 1,
    IsImplied = 2,
    IsInConflictClause = 4,
}

#[derive(Clone, Copy)]
pub struct LitFlags(u8);

impl LitFlags {

    #[inline]
    pub fn new() -> Self {
        LitFlags(LitFlag::Clear as u8)
    }
    
    #[inline]
    pub fn set(&mut self, flag: LitFlag) {
        self.0 |= flag as u8;
    }
    
    #[inline]
    pub fn unset(&mut self, flag: LitFlag) {
        self.0 ^= flag as u8;
    }
    
    #[inline]
    pub fn clear(&mut self) {
        self.0 = LitFlag::Clear as u8;
    }
    
    #[inline]
    pub fn is_set(&self, flag: LitFlag) -> bool {
        self.0 & (flag as u8) != 0
    }
}