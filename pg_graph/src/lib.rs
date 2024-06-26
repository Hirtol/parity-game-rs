#![feature(is_sorted)]

pub use datatypes::*;

mod datatypes;
pub mod explicit;
pub mod symbolic;
pub mod visualize;

#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;

    use pg_parser::parse_pg;

    use crate::explicit::ParityGame;
    use crate::explicit::solvers::SolverOutput;

    pub fn load_example(example: impl AsRef<str>) -> ParityGame {
        let input = std::fs::read_to_string(example_dir().join(example.as_ref())).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        ParityGame::new(pg).unwrap()
    }
    
    /// Load the given example and calculate an authoritative solution (using the explicit Zielonka solver)
    /// 
    /// The returned function can be used to compare a new solver output with the authoritative one.
    pub fn load_and_compare_example(example: impl AsRef<str>) -> (ParityGame, impl FnOnce(SolverOutput)) {
        let game_to_run = load_example(example);
        let mut solver = crate::explicit::solvers::zielonka::ZielonkaSolver::new(&game_to_run);
        let solution_author = solver.run();
        let compare = move |solution| {
            compare_solutions(solution_author, solution)
        };
        
        (game_to_run, compare)
    }
    
    pub fn compare_solutions(left: SolverOutput, right: SolverOutput) {
        assert_eq!(left.winners, right.winners, "Solutions of left and right solver don't match");

        println!("Solution: {:#?}", left.winners);
        
        if let (Some(strat_l), Some(strat_r)) = (left.strategy, right.strategy) {
            assert_eq!(strat_l, strat_r, "Strategies of left and right solver don't match");
        }
    }

    pub fn example_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("game_examples")
    }

    pub fn trivial_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 1;
0 1 1 0 "0";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    pub fn trivial_pg_2() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 2;
0 0 0 0,1 "0";
1 1 1 1,0 "1";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }
}
