use std::collections::HashMap;
use crate::ast::*;

// ── Tokenizer ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    LParen,
    RParen,
    Symbol(String),
}

fn tokenize(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_comment = false;

    for ch in source.chars() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
            }
            continue;
        }

        if ch == ';' {
            in_comment = true;
            // Flush any accumulated symbol before the comment
            if !current.is_empty() {
                tokens.push(Token::Symbol(current.clone()));
                current.clear();
            }
            continue;
        }

        if ch == '(' || ch == ')' {
            if !current.is_empty() {
                tokens.push(Token::Symbol(current.clone()));
                current.clear();
            }
            tokens.push(if ch == '(' {
                Token::LParen
            } else {
                Token::RParen
            });
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(Token::Symbol(current.clone()));
                current.clear();
            }
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        tokens.push(Token::Symbol(current));
    }

    tokens
}

// ── Recursive-descent parser ───────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Result<&Token, String> {
        let tok = self.tokens.get(self.pos).ok_or("Unexpected end of input")?;
        self.pos += 1;
        Ok(tok)
    }

    fn expect_symbol(&mut self) -> Result<String, String> {
        match self.advance()? {
            Token::Symbol(s) => Ok(s.clone()),
            other => Err(format!("Expected symbol, got {:?}", other)),
        }
    }

    fn expect_lparen(&mut self) -> Result<(), String> {
        match self.advance()? {
            Token::LParen => Ok(()),
            other => Err(format!("Expected '(', got {:?}", other)),
        }
    }

    fn expect_rparen(&mut self) -> Result<(), String> {
        match self.advance()? {
            Token::RParen => Ok(()),
            other => Err(format!("Expected ')', got {:?}", other)),
        }
    }

    /// Parse a simple list: (keyword val1 val2 ...)  → returns all items
    fn parse_list(&mut self) -> Result<Vec<String>, String> {
        self.expect_lparen()?;
        let mut items = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RParen) | None => break,
                _ => items.push(self.expect_symbol()?),
            }
        }
        self.expect_rparen()?;
        Ok(items)
    }

    /// Parse a node definition:
    ///   (node name Opcode (wait src slot) ... (send target slot) ...)
    fn parse_node(&mut self) -> Result<NodeBlueprint, String> {
        self.expect_lparen()?;

        let keyword = self.expect_symbol()?;
        if keyword != "node" {
            return Err(format!("Expected 'node', got '{}'", keyword));
        }

        let name = self.expect_symbol()?;
        let opcode_str = self.expect_symbol()?;

        let opcode = match opcode_str.as_str() {
            "Add" => Opcode::Add,
            "Mul" => Opcode::Mul,
            "CmpGeZ" => Opcode::CmpGeZ,
            "Switch" => Opcode::Switch,
            "Merge" => Opcode::Merge,
            _ => return Err(format!("Unknown opcode: {}", opcode_str)),
        };

        let mut wait_sources = HashMap::new();
        let mut destinations = Vec::new();

        // Parse (wait …), (send …), (send_true …), (send_false …) clauses
        loop {
            match self.peek() {
                Some(Token::LParen) => { /* continue */ }
                Some(Token::RParen) | None => break,
                _ => {
                    // Unexpected token at node level
                    let tok = self.advance()?;
                    return Err(format!("Unexpected token in node body: {:?}", tok));
                }
            }

            self.expect_lparen()?;
            let directive = self.expect_symbol()?;

            match directive.as_str() {
                "wait" => {
                    let source = self.expect_symbol()?;
                    let slot_str = self.expect_symbol()?;
                    let slot: usize = slot_str
                        .parse()
                        .map_err(|_| format!("Invalid slot: {}", slot_str))?;
                    wait_sources.insert(source, slot);
                }
                "send" => {
                    let target = self.expect_symbol()?;
                    let slot_str = self.expect_symbol()?;
                    let slot: usize = slot_str
                        .parse()
                        .map_err(|_| format!("Invalid slot: {}", slot_str))?;
                    destinations.push(Destination::Always { target, slot });
                }
                "send_true" => {
                    let target = self.expect_symbol()?;
                    let slot_str = self.expect_symbol()?;
                    let slot: usize = slot_str
                        .parse()
                        .map_err(|_| format!("Invalid slot: {}", slot_str))?;
                    destinations.push(Destination::IfTrue { target, slot });
                }
                "send_false" => {
                    let target = self.expect_symbol()?;
                    let slot_str = self.expect_symbol()?;
                    let slot: usize = slot_str
                        .parse()
                        .map_err(|_| format!("Invalid slot: {}", slot_str))?;
                    destinations.push(Destination::IfFalse { target, slot });
                }
                _ => return Err(format!("Unknown directive in node: {}", directive)),
            }
            self.expect_rparen()?;
        }

        self.expect_rparen()?; // close (node …)

        let inputs_required = wait_sources.len();

        Ok(NodeBlueprint {
            name,
            opcode,
            inputs_required,
            wait_sources,
            destinations,
        })
    }

    fn parse_program(&mut self) -> Result<Program, String> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut nodes = HashMap::new();

        while self.peek().is_some() {
            match self.peek().unwrap() {
                Token::RParen => {
                    return Err("Unexpected ')' at top level".to_string());
                }
                Token::LParen => {
                    // Peek inside to decide what we're parsing
                    let saved = self.pos;
                    self.pos += 1; // skip '('
                    match self.peek() {
                        Some(Token::Symbol(kw)) if kw == "node" => {
                            self.pos = saved;
                            let node = self.parse_node()?;
                            nodes.insert(node.name.clone(), node);
                        }
                        Some(Token::Symbol(kw)) => {
                            let kw = kw.clone();
                            self.pos = saved;
                            let items = self.parse_list()?;
                            if items.is_empty() {
                                return Err(format!("Empty ({}) block", kw));
                            }
                            match kw.as_str() {
                                "input" => inputs = items.into_iter().skip(1).collect(),
                                "output" => outputs = items.into_iter().skip(1).collect(),
                                _ => {
                                    return Err(format!(
                                        "Unknown top-level form: {}",
                                        kw
                                    ))
                                }
                            }
                        }
                        _ => {
                            self.pos = saved;
                            self.parse_list()?; // skip and ignore
                        }
                    }
                }
                Token::Symbol(s) => {
                    let sym = s.clone();
                    self.pos += 1;
                    return Err(format!("Unexpected symbol at top level: {}", sym));
                }
            }
        }

        Ok(Program {
            inputs,
            outputs,
            nodes,
        })
    }
}

pub fn parse(source: &str) -> Result<Program, String> {
    let tokens = tokenize(source);
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

// ── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_simple_program() {
        let src = "(input a b)\n(node mul_00 Mul (wait a 0) (wait b 1) (send add_00 0))";
        let tokens = tokenize(src);
        let symbols: Vec<String> = tokens
            .iter()
            .filter_map(|t| match t {
                Token::Symbol(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(symbols, vec!["input", "a", "b", "node", "mul_00", "Mul", "wait", "a", "0", "wait", "b", "1", "send", "add_00", "0"]);
    }

    #[test]
    fn parse_basic_add() {
        let src = "(input a b)\n(output r)\n(node add_00 Add (wait a 0) (wait b 1) (send r 0))\n";
        let prog = parse(src).unwrap();
        assert_eq!(prog.inputs, vec!["a", "b"]);
        assert_eq!(prog.outputs, vec!["r"]);
        assert_eq!(prog.nodes.len(), 1);
        let node = prog.nodes.get("add_00").unwrap();
        assert_eq!(node.opcode, Opcode::Add);
        assert_eq!(node.inputs_required, 2);
        assert_eq!(node.destinations.len(), 1);
    }

    #[test]
    fn parse_switch_node() {
        let src = "(input data cond)\n(output a b)\n(node sw Switch (wait data 0) (wait cond 1) (send_true a 0) (send_false b 0))\n";
        let prog = parse(src).unwrap();
        let node = prog.nodes.get("sw").unwrap();
        assert_eq!(node.opcode, Opcode::Switch);
        assert_eq!(node.destinations.len(), 2);
        assert!(matches!(node.destinations[0], Destination::IfTrue { .. }));
        assert!(matches!(node.destinations[1], Destination::IfFalse { .. }));
    }
}
