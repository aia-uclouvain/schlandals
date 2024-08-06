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

use schlandals::Args;
use schlandals::Command;
use clap::Parser;

fn main() {
    let args = Args::parse();
    match args.subcommand {
        Some(Command::Compile { .. }) => { schlandals::compile(args); },
        Some(Command::Learn { .. }) => { schlandals::learn(args); },
        None => {schlandals::search(args); },
    };
}
