use std::collections::{HashMap, VecDeque};
use crate::ast::{Destination, Opcode, Program};

#[derive(Debug, Clone)]
struct RuntimeNode {
    opcode: Opcode,
    inputs_required: usize,
    inputs_received: usize,
    operands: [Option<f32>; 2],
    destinations: Vec<Destination>,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub target: String,
    pub slot: usize,
    pub value: f32,
}

pub struct VirtualDataflowCPU {
    nodes: HashMap<String, RuntimeNode>,
    /// Maps source names (inputs or node outputs) to consumer (node, slot) pairs
    input_map: HashMap<String, Vec<(String, usize)>>,
    token_queue: VecDeque<Token>,
    ready_pool: VecDeque<String>,
    outputs: HashMap<String, Option<f32>>,
    total_tokens_processed: u64,
    max_queue_depth: usize,
}

impl VirtualDataflowCPU {
    pub fn new(program: &Program) -> Self {
        let mut nodes = HashMap::new();
        let mut input_map: HashMap<String, Vec<(String, usize)>> = HashMap::new();

        for (name, bp) in &program.nodes {
            // Merge only needs 1 token to fire
            let inputs_required = if bp.opcode == Opcode::Merge {
                1
            } else {
                bp.inputs_required
            };

            nodes.insert(
                name.clone(),
                RuntimeNode {
                    opcode: bp.opcode,
                    inputs_required,
                    inputs_received: 0,
                    operands: [None; 2],
                    destinations: bp.destinations.clone(),
                },
            );

            // Build reverse index: source_name → [(consumer_node, slot)]
            for (source_name, &slot) in &bp.wait_sources {
                input_map
                    .entry(source_name.clone())
                    .or_default()
                    .push((name.clone(), slot));
            }
        }

        Self {
            nodes,
            input_map,
            token_queue: VecDeque::new(),
            ready_pool: VecDeque::new(),
            outputs: program.outputs.iter().map(|o| (o.clone(), None)).collect(),
            total_tokens_processed: 0,
            max_queue_depth: 0,
        }
    }

    /// Inject an external input token, fanning out to all consumers.
    pub fn inject_input(&mut self, input_name: &str, value: f32) {
        if let Some(consumers) = self.input_map.get(input_name) {
            for (node_name, slot) in consumers {
                self.token_queue.push_back(Token {
                    target: node_name.clone(),
                    slot: *slot,
                    value,
                });
            }
        }
    }

    /// Inject a pre-built token directly into the queue (for node outputs).
    fn inject_token(&mut self, token: Token) {
        self.token_queue.push_back(token);
    }

    /// Execute one micro-step: process one token OR fire one ready node.
    /// Returns true if there is still work remaining.
    pub fn step(&mut self) -> bool {
        // Track peak queue depth BEFORE any early returns
        let current_depth = self.token_queue.len() + self.ready_pool.len();
        if current_depth > self.max_queue_depth {
            self.max_queue_depth = current_depth;
        }

        // ── Phase 1: Process an in-flight token ──
        if let Some(token) = self.token_queue.pop_front() {
            self.total_tokens_processed += 1;

            // Output sink?
            if self.outputs.contains_key(&token.target) {
                self.outputs.insert(token.target.clone(), Some(token.value));
                return !self.token_queue.is_empty() || !self.ready_pool.is_empty();
            }

            // Deposit into matching store
            if let Some(node) = self.nodes.get_mut(&token.target) {
                if token.slot < 2 && node.operands[token.slot].is_none() {
                    node.operands[token.slot] = Some(token.value);
                    node.inputs_received += 1;
                }

                // Use == to avoid duplicate entries when redundant tokens arrive
                if node.inputs_received == node.inputs_required {
                    self.ready_pool.push_back(token.target.clone());
                }
            }
        }

        // ── Phase 2: Fire a ready node ──
        if let Some(ready_name) = self.ready_pool.pop_front() {
            let result;
            let mut branch_condition = true;
            let destinations: Vec<Destination>;

            {
                let Some(node) = self.nodes.get_mut(&ready_name) else {
                    // Shouldn't happen, but defensive: skip phantom ready entries
                    return !self.token_queue.is_empty() || !self.ready_pool.is_empty();
                };

                result = match node.opcode {
                    Opcode::Add => {
                        node.operands[0].unwrap_or(0.0) + node.operands[1].unwrap_or(0.0)
                    }
                    Opcode::Mul => {
                        node.operands[0].unwrap_or(0.0) * node.operands[1].unwrap_or(0.0)
                    }
                    Opcode::Switch => {
                        // Slot 1 is the control boolean; slot 0 is the data to route
                        branch_condition = node.operands[1].unwrap_or(0.0) != 0.0;
                        node.operands[0].unwrap_or(0.0)
                    }
                    Opcode::Merge => node.operands[0].or(node.operands[1]).unwrap_or(0.0),
                    Opcode::CmpGeZ => {
                        let val = node.operands[0].unwrap_or(0.0);
                        if val >= 0.0 { 1.0 } else { 0.0 }
                    }
                };

                destinations = node.destinations.clone();

                // Flush matching-store state so node can re-fire
                node.inputs_received = 0;
                node.operands = [None; 2];
            }

            // Broadcast results according to routing rules
            for dest in &destinations {
                let should_send = match dest {
                    Destination::Always { .. } => true,
                    Destination::IfTrue { .. } => branch_condition,
                    Destination::IfFalse { .. } => !branch_condition,
                };

                if should_send {
                    let (target, slot) = match dest {
                        Destination::Always { target, slot }
                        | Destination::IfTrue { target, slot }
                        | Destination::IfFalse { target, slot } => (target.clone(), *slot),
                    };

                    self.inject_token(Token {
                        target,
                        slot,
                        value: result,
                    });
                }
            }
        }

        // Track peak queue depth for telemetry
        let current_depth = self.token_queue.len() + self.ready_pool.len();
        if current_depth > self.max_queue_depth {
            self.max_queue_depth = current_depth;
        }

        !self.token_queue.is_empty() || !self.ready_pool.is_empty()
    }

    /// Run until quiescence (no more tokens or ready nodes).
    /// Returns the number of steps taken.
    pub fn run_to_completion(&mut self) -> u64 {
        let mut steps = 0u64;
        while self.step() {
            steps += 1;
            if steps > 1_000_000 {
                eprintln!("WARNING: VM exceeded 1M steps — possible livelock");
                break;
            }
        }
        steps
    }

    #[allow(dead_code)]
    pub fn get_output(&self, name: &str) -> Option<f32> {
        self.outputs.get(name).copied().flatten()
    }

    #[allow(dead_code)]
    pub fn all_outputs(&self) -> &HashMap<String, Option<f32>> {
        &self.outputs
    }

    #[allow(dead_code)]
    pub fn token_count(&self) -> u64 {
        self.total_tokens_processed
    }

    pub fn max_queue_depth(&self) -> usize {
        self.max_queue_depth
    }
}
