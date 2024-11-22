use clap::Parser;
use peak_alloc::PeakAlloc;
use tracing_subscriber::util::SubscriberInitExt;

use crate::args::SubCommands;

#[cfg(not(feature = "dhat-heap"))]
#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

mod args;
mod trace;

#[profiling::function]
fn main() -> eyre::Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    let args = args::ClapArgs::parse();

    trace::create_subscriber("DEBUG,pg_graph=DEBUG").init();

    let now = std::time::Instant::now();

    match args.commands {
        SubCommands::Solve(solv) => {
            solv.run()?;
        }
        SubCommands::Convert(conv) => {
            conv.run()?;
        }
        SubCommands::Bench(bench) => {
            bench.run()?;
        }
        SubCommands::Test(test) => {
            test.run()?;
        }
    }

    #[cfg(not(feature = "dhat-heap"))]
    {
        tracing::info!(
            "Runtime: {:.2?} - Peak Memory Usage: {} MB",
            now.elapsed(),
            PEAK_ALLOC.peak_usage_as_mb()
        );
    }
    #[cfg(feature = "dhat-heap")]
    {
        tracing::info!(
            "Runtime: {:.2?} - Peak Memory Usage: {} MB",
            now.elapsed(),
            "UNKNOWN"
        );
    }

    Ok(())
}
