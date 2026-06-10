mod ast;
mod parser;
mod vm;

use vm::VirtualDataflowCPU;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: dfasm <file.dfasm> [--input name=value ...]");
        eprintln!("  If no --input values are given, defaults for 2×2 matmul are used.");
        eprintln!("  Always prints METRICS: CYCLES=X, MAX_QUEUE=Y on the last line.");
        return Ok(());
    }

    // ── Parse arguments ─────────────────────────────────
    let dfasm_path = &args[1];
    let mut named_inputs: std::collections::HashMap<String, f32> =
        std::collections::HashMap::new();

    let mut i = 2;
    while i < args.len() {
        if args[i] == "--input" && i + 1 < args.len() {
            let pair = &args[i + 1];
            if let Some(eq_pos) = pair.find('=') {
                let name = pair[..eq_pos].to_string();
                let val: f32 = pair[eq_pos + 1..]
                    .parse()
                    .map_err(|_| format!("Invalid float value in --input: {}", pair))?;
                named_inputs.insert(name, val);
            }
            i += 2;
        } else {
            i += 1;
        }
    }

    let source =
        std::fs::read_to_string(dfasm_path)
            .map_err(|e| format!("Cannot read {}: {}", dfasm_path, e))?;

    let program = parser::parse(&source)?;

    let mut cpu = VirtualDataflowCPU::new(&program);

    // ── Inject inputs ──────────────────────────────────
    if named_inputs.is_empty() {
        // Default matmul values when no --input flags given
        let defaults = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (i, name) in program.inputs.iter().enumerate() {
            let value = defaults.get(i).copied().unwrap_or(0.0);
            cpu.inject_input(name, value);
        }
    } else {
        for name in &program.inputs {
            let value = named_inputs.get(name).copied().unwrap_or(0.0);
            cpu.inject_input(name, value);
        }
    }

    // ── Run ────────────────────────────────────────────
    let steps = cpu.run_to_completion();

    // ── Output values ──
    let output_str: Vec<String> = program
        .outputs
        .iter()
        .map(|name| {
            let val = cpu.get_output(name).unwrap_or(f32::NAN);
            format!("{}={}", name, val)
        })
        .collect();
    println!("OUTPUTS: {}", output_str.join(" "));

    // ── Machine-parseable metrics (always last line) ──
    println!(
        "METRICS: CYCLES={}, MAX_QUEUE={}",
        steps,
        cpu.max_queue_depth()
    );

    Ok(())
}

// ── Integration tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2_correctness() {
        let src = include_str!("../examples/matmul_2x2.dfasm");
        let program = parser::parse(src).expect("parse matmul");

        let mut cpu = VirtualDataflowCPU::new(&program);

        // A = [[1,2],[3,4]]   B = [[5,6],[7,8]]
        // Expected: 19, 22, 43, 50
        let inputs: [(&str, f32); 8] = [
            ("a00", 1.0),
            ("a01", 2.0),
            ("a10", 3.0),
            ("a11", 4.0),
            ("b00", 5.0),
            ("b01", 6.0),
            ("b10", 7.0),
            ("b11", 8.0),
        ];

        for (name, val) in &inputs {
            cpu.inject_input(name, *val);
        }

        let _steps = cpu.run_to_completion();

        assert_eq!(cpu.get_output("c00"), Some(19.0));
        assert_eq!(cpu.get_output("c01"), Some(22.0));
        assert_eq!(cpu.get_output("c10"), Some(43.0));
        assert_eq!(cpu.get_output("c11"), Some(50.0));
    }

    #[test]
    fn switch_route_true() {
        let src = "
(input data cond)
(output route_true route_false)
(node sw Switch
    (wait data 0)
    (wait cond 1)
    (send_true route_true 0)
    (send_false route_false 0)
)
";
        let program = parser::parse(src).expect("parse switch");
        let mut cpu = VirtualDataflowCPU::new(&program);
        cpu.inject_input("data", 42.0);
        cpu.inject_input("cond", 1.0); // true → route to route_true
        cpu.run_to_completion();

        assert_eq!(cpu.get_output("route_true"), Some(42.0));
        assert_eq!(cpu.get_output("route_false"), None);
    }

    #[test]
    fn switch_route_false() {
        let src = "
(input data cond)
(output route_true route_false)
(node sw Switch
    (wait data 0)
    (wait cond 1)
    (send_true route_true 0)
    (send_false route_false 0)
)
";
        let program = parser::parse(src).expect("parse switch");
        let mut cpu = VirtualDataflowCPU::new(&program);
        cpu.inject_input("data", 99.0);
        cpu.inject_input("cond", 0.0); // false → route to route_false
        cpu.run_to_completion();

        assert_eq!(cpu.get_output("route_true"), None);
        assert_eq!(cpu.get_output("route_false"), Some(99.0));
    }

    #[test]
    fn merge_first_arrival() {
        let src = "
(input a b)
(output r)
(node m Merge (wait a 0) (wait b 1) (send r 0))
";
        let program = parser::parse(src).unwrap();
        let mut cpu = VirtualDataflowCPU::new(&program);
        cpu.inject_input("a", 7.0);
        cpu.run_to_completion();
        assert_eq!(cpu.get_output("r"), Some(7.0));
    }
}
