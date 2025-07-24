//! Time compatibility layer for WASM and native

use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant {
    millis: f64,
}

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        let millis = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        Instant { millis }
    }
    
    pub fn elapsed(&self) -> Duration {
        let now = Self::now();
        let elapsed_millis = now.millis - self.millis;
        Duration::from_millis(elapsed_millis as u64)
    }
}