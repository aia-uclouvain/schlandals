use std::fmt;

/// Implements a bunch of statistics that are collected during the search
#[derive(Default)]
pub struct Statistics<const B: bool> {
    cache_miss: usize,
    cache_access: usize,
    number_or_nodes: usize,
    number_and_nodes: usize,
    total_and_decompositions: usize,
    number_unsat: usize,
}

impl<const B: bool> Statistics<B> {
    pub fn cache_miss(&mut self) {
        if B {
            self.cache_miss += 1;
        }
    }

    pub fn cache_access(&mut self) {
        if B {
            self.cache_access += 1;
        }
    }

    pub fn or_node(&mut self) {
        if B {
            self.number_or_nodes += 1;
        }
    }

    pub fn and_node(&mut self) {
        if B {
            self.number_and_nodes += 1;
        }
    }

    pub fn decomposition(&mut self, number_components: usize) {
        if B {
            self.total_and_decompositions += number_components;
            if number_components > 1 {
                self.and_node();
            }
        }
    }
    
    pub fn unsat(&mut self) {
       if B {
            self.number_unsat += 1;
       } 
    }
    
    pub fn print(&self) {
        if B {
            println!("{}", self);
        }
    }
}

impl<const B: bool> fmt::Display for Statistics<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if B {
            let cache_hit_percentages = 100f64 - (self.cache_miss as f64 / self.cache_access as f64) * 100.0;
            let avg_decomposition = if self.number_and_nodes > 1 {
                (self.total_and_decompositions as f64) / (self.number_and_nodes as f64)
            } else {
                1.0
            };
            writeln!(f,
                "cache_hit {:.3} | OR nodes {} | AND nodes {} | avg decomposition {} | #UNSAT {}",
                cache_hit_percentages,
                self.number_or_nodes,
                self.number_and_nodes,
                avg_decomposition,
                self.number_unsat)
        } else {
            write!(f, "")
        }
    }
}
