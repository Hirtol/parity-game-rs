use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;
use simplelog::LevelFilter;
use pg_graph::{Owner, ParityGame};
use pg_graph::solvers::register_game::RegisterGame;
use pg_graph::solvers::small_progress::SmallProgressSolver;
use pg_parser::parse_pg;
use crate::args::SubCommands;

mod args;
mod utils;

fn main() -> eyre::Result<()> {
    let args = args::ClapArgs::parse();
    
    let now = std::time::Instant::now();
    
    match args.commands {
        SubCommands::Solve(solv) => {
            solv.run()?;
        }
        SubCommands::Convert(conv) => {
            conv.run()?;
        }
    }
    
    println!("Runtime: {:?}", now.elapsed());
    
    let input = std::fs::read_to_string(example_dir().join("amba_decomposed_arbiter_6.tlsf.ehoa.pg")).unwrap();
    let pg = parse_pg(&mut input.as_str()).unwrap();
    let game = ParityGame::new(pg).unwrap();
    println!("Memory Usage: {:#?}", memory_stats::memory_stats());
    std::fs::write("game.mmd", game.to_mermaid())?;

    let now = Instant::now();
    let register_game = RegisterGame::construct(game, 1, Owner::Even);
    println!("Register conversion Took: {:?}", now.elapsed());
    let now = Instant::now();

    std::fs::write("register_game.mmd", register_game.to_mermaid())?;
    println!("Mermaid Took: {:?}", now.elapsed());

    let now = Instant::now();
    let game = register_game.to_game()?;
    std::fs::write("register_game_to_pg.mmd", game.to_mermaid())?;
    std::fs::write("register_game_to_pg.pg", game.to_pg())?;
    println!("Game conversion Took: {:?}", now.elapsed());

    let mut solver = SmallProgressSolver::new(game);

    let now = Instant::now();
    let solution = solver.run();
    let projected_solution = register_game.project_winners_original(&solution);

    println!("Prog Count: {}", solver.prog_count);
    println!("Took: {:?}", now.elapsed());
    // println!("Solution: {:?}", solution);
    // println!("Projected Solution: {:?}", projected_solution);

    Ok(())
}

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("game_examples")
}
