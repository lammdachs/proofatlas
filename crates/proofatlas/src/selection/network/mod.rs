//! Network communication for remote scoring

pub mod protocol;
#[cfg(unix)]
pub mod remote;
#[cfg(feature = "ml")]
pub mod server;
