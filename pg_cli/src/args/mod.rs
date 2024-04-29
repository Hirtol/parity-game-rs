use convert::ConvertCommand;
use solve::SolveCommand;
use crate::args::bench::BenchCommand;

pub mod solve;
pub mod convert;
pub mod bench;

#[derive(clap::Parser, Debug)]
#[clap(version, about)]
pub struct ClapArgs {
    #[clap(subcommand)]
    pub commands: SubCommands,
}

#[derive(clap::Subcommand, Debug)]
pub enum SubCommands {
    /// Solve a given parity game
    /// 
    /// Will use the SPM algorithm by default.
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "s")]
    Solve(SolveCommand),
    /// Convert a given parity game to various formats
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "c")]
    Convert(ConvertCommand),
    /// Run the benchmark for the paper.
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "b")]
    Bench(BenchCommand),
}