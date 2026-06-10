use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opcode {
    Add,
    Mul,
    Switch,
    Merge,
    CmpGeZ,
}

#[derive(Debug, Clone)]
pub enum Destination {
    Always { target: String, slot: usize },
    IfTrue { target: String, slot: usize },
    IfFalse { target: String, slot: usize },
}

#[derive(Debug, Clone)]
pub struct NodeBlueprint {
    pub name: String,
    pub opcode: Opcode,
    pub inputs_required: usize,
    /// source name → operand slot
    pub wait_sources: HashMap<String, usize>,
    pub destinations: Vec<Destination>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub nodes: HashMap<String, NodeBlueprint>,
}
