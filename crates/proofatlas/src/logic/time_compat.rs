//! Time compatibility layer for WASM and native

#[allow(unused_imports)]
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

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
        // Use js_sys::global() to get performance.now() — works in both Window and Worker
        let millis = js_sys::Reflect::get(&js_sys::global(), &"performance".into())
            .ok()
            .and_then(|perf| {
                let perf: web_sys::Performance = perf.dyn_into().ok()?;
                Some(perf.now())
            })
            .unwrap_or(0.0);
        Instant { millis }
    }

    pub fn elapsed(&self) -> Duration {
        let now = Self::now();
        let elapsed_millis = now.millis - self.millis;
        Duration::from_millis(elapsed_millis as u64)
    }
}
