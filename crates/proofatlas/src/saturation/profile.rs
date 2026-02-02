//! Structured profiling for the saturation loop
//!
//! Collects timing and counting data during saturation when enabled via
//! `SaturationConfig::enable_profiling`. Zero overhead when disabled — all
//! instrumentation is gated on `Option::None`.

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use std::time::Duration;

fn secs(d: &Duration) -> f64 {
    d.as_secs_f64()
}

/// Profiling data collected during saturation.
///
/// All `Duration` fields are serialized as `f64` seconds.
#[derive(Debug, Clone, Default)]
pub struct SaturationProfile {
    // Top-level phase timings
    pub total_time: Duration,
    pub forward_simplify_time: Duration,
    pub select_given_time: Duration,
    pub generate_inferences_time: Duration,
    pub add_inferences_time: Duration,

    // Forward simplification sub-phases
    pub forward_demod_time: Duration,
    pub forward_subsumption_time: Duration,
    pub backward_subsumption_time: Duration,
    pub backward_demod_time: Duration,

    // Inference rule counts
    pub resolution_count: usize,
    pub superposition_count: usize,
    pub factoring_count: usize,
    pub equality_resolution_count: usize,
    pub equality_factoring_count: usize,
    pub demodulation_count: usize,

    // Inference rule timings
    pub resolution_time: Duration,
    pub superposition_time: Duration,
    pub factoring_time: Duration,
    pub equality_resolution_time: Duration,
    pub equality_factoring_time: Duration,

    // Aggregate counters
    pub iterations: usize,
    pub clauses_generated: usize,
    pub clauses_added: usize,
    pub clauses_subsumed_forward: usize,
    pub clauses_subsumed_backward: usize,
    pub clauses_demodulated_forward: usize,
    pub clauses_demodulated_backward: usize,
    pub tautologies_deleted: usize,
    pub max_unprocessed_size: usize,
    pub max_processed_size: usize,

    // Selector stats (filled post-saturation)
    pub selector_name: String,
    pub selector_cache_hits: usize,
    pub selector_cache_misses: usize,
    pub selector_embed_time: Duration,
    pub selector_score_time: Duration,
}

impl Serialize for SaturationProfile {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("SaturationProfile", 32)?;

        // Top-level phase timings
        s.serialize_field("total_time", &secs(&self.total_time))?;
        s.serialize_field("forward_simplify_time", &secs(&self.forward_simplify_time))?;
        s.serialize_field("select_given_time", &secs(&self.select_given_time))?;
        s.serialize_field("generate_inferences_time", &secs(&self.generate_inferences_time))?;
        s.serialize_field("add_inferences_time", &secs(&self.add_inferences_time))?;

        // Forward simplification sub-phases
        s.serialize_field("forward_demod_time", &secs(&self.forward_demod_time))?;
        s.serialize_field("forward_subsumption_time", &secs(&self.forward_subsumption_time))?;
        s.serialize_field("backward_subsumption_time", &secs(&self.backward_subsumption_time))?;
        s.serialize_field("backward_demod_time", &secs(&self.backward_demod_time))?;

        // Inference rule counts
        s.serialize_field("resolution_count", &self.resolution_count)?;
        s.serialize_field("superposition_count", &self.superposition_count)?;
        s.serialize_field("factoring_count", &self.factoring_count)?;
        s.serialize_field("equality_resolution_count", &self.equality_resolution_count)?;
        s.serialize_field("equality_factoring_count", &self.equality_factoring_count)?;
        s.serialize_field("demodulation_count", &self.demodulation_count)?;

        // Inference rule timings
        s.serialize_field("resolution_time", &secs(&self.resolution_time))?;
        s.serialize_field("superposition_time", &secs(&self.superposition_time))?;
        s.serialize_field("factoring_time", &secs(&self.factoring_time))?;
        s.serialize_field("equality_resolution_time", &secs(&self.equality_resolution_time))?;
        s.serialize_field("equality_factoring_time", &secs(&self.equality_factoring_time))?;

        // Aggregate counters
        s.serialize_field("iterations", &self.iterations)?;
        s.serialize_field("clauses_generated", &self.clauses_generated)?;
        s.serialize_field("clauses_added", &self.clauses_added)?;
        s.serialize_field("clauses_subsumed_forward", &self.clauses_subsumed_forward)?;
        s.serialize_field("clauses_subsumed_backward", &self.clauses_subsumed_backward)?;
        s.serialize_field("clauses_demodulated_forward", &self.clauses_demodulated_forward)?;
        s.serialize_field("clauses_demodulated_backward", &self.clauses_demodulated_backward)?;
        s.serialize_field("tautologies_deleted", &self.tautologies_deleted)?;
        s.serialize_field("max_unprocessed_size", &self.max_unprocessed_size)?;
        s.serialize_field("max_processed_size", &self.max_processed_size)?;

        // Selector stats
        s.serialize_field("selector_name", &self.selector_name)?;
        s.serialize_field("selector_cache_hits", &self.selector_cache_hits)?;
        s.serialize_field("selector_cache_misses", &self.selector_cache_misses)?;
        s.serialize_field("selector_embed_time", &secs(&self.selector_embed_time))?;
        s.serialize_field("selector_score_time", &secs(&self.selector_score_time))?;

        s.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile_serializes() {
        let profile = SaturationProfile::default();
        let json = serde_json::to_string(&profile).unwrap();
        assert!(json.contains("\"total_time\":0.0"));
        assert!(json.contains("\"iterations\":0"));
        assert!(json.contains("\"selector_name\":\"\""));
    }

    #[test]
    fn test_profile_with_values_serializes() {
        let mut profile = SaturationProfile::default();
        profile.total_time = Duration::from_millis(1500);
        profile.iterations = 42;
        profile.clauses_generated = 100;
        profile.selector_name = "AgeWeight".to_string();

        let json = serde_json::to_string(&profile).unwrap();
        assert!(json.contains("\"total_time\":1.5"));
        assert!(json.contains("\"iterations\":42"));
        assert!(json.contains("\"clauses_generated\":100"));
        assert!(json.contains("\"selector_name\":\"AgeWeight\""));
    }

    #[test]
    fn test_profile_deserializes_as_value() {
        let mut profile = SaturationProfile::default();
        profile.total_time = Duration::from_secs(2);
        profile.resolution_count = 10;

        let json = serde_json::to_string(&profile).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["total_time"], 2.0);
        assert_eq!(value["resolution_count"], 10);
    }
}
