use crate::args::bench::BenchCommand;
use crate::args::testing::TestingCommand;
use convert::ConvertCommand;
use solve::SolveCommand;

pub mod bench;
pub mod convert;
pub mod solve;
pub mod testing;

#[derive(clap::Parser, Debug)]
#[clap(version, about)]
pub struct ClapArgs {
    #[clap(subcommand)]
    pub commands: SubCommands,
    /// Use a simple trace format for piping the output to files.
    #[clap(long, global = true)]
    pub simple_trace: bool,
}

#[derive(clap::Subcommand, Debug)]
pub enum SubCommands {
    /// Solve a given parity game
    ///
    /// Will use the Zielonka's algorithm by default
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "s")]
    Solve(SolveCommand),
    /// Convert a given parity game to various formats
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "c")]
    Convert(ConvertCommand),
    /// Run the benchmark for the paper
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "b")]
    Bench(BenchCommand),
    /// Random tests for development, don't use
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "t")]
    Test(TestingCommand),
}
