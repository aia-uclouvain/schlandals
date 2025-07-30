pub mod ac;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(pub usize);

pub struct Edge {
    to: NodeIndex,
    next: Option<EdgeIndex>,
}

impl Edge {
    pub fn to(&self) -> NodeIndex {
        self.to
    }

    pub fn next(&self) -> Option<EdgeIndex> {
        self.next
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeIndex(usize);
