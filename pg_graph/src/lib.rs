#![feature(is_sorted)]

pub use datatypes::*;
pub use petgraph;

mod datatypes;
pub mod explicit;
pub mod symbolic;
pub mod visualize;

#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;

    use pg_parser::parse_pg;

    use crate::explicit::solvers::SolverOutput;
    use crate::explicit::{ParityGame, ParityGameBuilder};

    pub fn load_example(example: impl AsRef<str>) -> ParityGame {
        let input = std::fs::read_to_string(example_dir().join(example.as_ref())).unwrap();
        parse_pg_from_str(input)
    }
    
    pub fn parse_pg_from_str(game: impl AsRef<str>) -> ParityGame {
        let mut builder = ParityGameBuilder::new();
        parse_pg(&mut game.as_ref(), &mut builder).unwrap();
        
        builder.build()
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

    pub fn trivial_pg() -> ParityGame {
        let pg = r#"parity 1;
0 1 1 0 "0";"#;
        parse_pg_from_str(pg)
    }

    pub fn trivial_pg_2() -> ParityGame {
        let pg = r#"parity 2;
0 0 0 0,1 "0";
1 1 1 1,0 "1";"#;
        parse_pg_from_str(pg)
    }
}
