//! Convenience macros for _very_ verbose logging of the solvers.
//! Kept separate to allow more complex logging

#[macro_export]
macro_rules! trace {
    ($($stuff:tt)* ) => {
        #[cfg(feature = "verbose")]
        tracing::trace!($($stuff)*);
    }
}

#[macro_export]
macro_rules! debug {
    ($($stuff:tt)* ) => {
        #[cfg(feature = "verbose")]
        tracing::debug!($($stuff)*);
    }
}

#[macro_export]
macro_rules! info {
    ($($stuff:tt)* ) => {
        #[cfg(feature = "verbose")]
        tracing::info!($($stuff)*);
    }
}

#[macro_export]
macro_rules! warn {
    ($($stuff:tt)* ) => {
        #[cfg(feature = "verbose")]
        tracing::warn!($($stuff)*);
    }
}

#[macro_export]
macro_rules! error {
    ($($stuff:tt)* ) => {
        #[cfg(feature = "verbose")]
        tracing::error!($($stuff)*);
    }
}
