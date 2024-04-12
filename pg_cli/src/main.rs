use clap::Parser;
use peak_alloc::PeakAlloc;
use tracing_subscriber::util::SubscriberInitExt;

use crate::args::SubCommands;

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

mod args;
mod utils;
mod trace;

#[profiling::function]
fn main() -> eyre::Result<()> {
    let args = args::ClapArgs::parse();
    
    trace::create_subscriber("DEBUG,pg_graph=INFO").init();
    
    let now = std::time::Instant::now();
    
    match args.commands {
        SubCommands::Solve(solv) => {
            solv.run()?;
        }
        SubCommands::Convert(conv) => {
            conv.run()?;
        }
    }
    
    tracing::info!("Runtime: {:.2?} - Peak Memory Usage: {} MB", now.elapsed(), PEAK_ALLOC.peak_usage_as_mb());

    Ok(())
}
