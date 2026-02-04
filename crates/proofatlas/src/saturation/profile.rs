//! Structured profiling for the saturation loop
//!
//! Collects timing and counting data during saturation when enabled via
//! `SaturationConfig::enable_profiling`. Zero overhead when disabled — all
//! instrumentation is gated on `Option::None`.

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use std::collections::HashMap;
use std::time::Duration;

fn secs(d: &Duration) -> f64 {
    d.as_secs_f64()
}

/// Statistics for a generating inference rule.
#[derive(Debug, Clone, Default)]
pub struct RuleStats {
    pub count: usize,
    pub time: Duration,
}

impl Serialize for RuleStats {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("RuleStats", 2)?;
        s.serialize_field("count", &self.count)?;
        s.serialize_field("time", &secs(&self.time))?;
        s.end()
    }
}

/// Statistics for a simplification rule (with forward and backward phases).
#[derive(Debug, Clone, Default)]
pub struct SimplificationStats {
    pub forward_count: usize,
    pub forward_time: Duration,
    pub backward_count: usize,
    pub backward_time: Duration,
    /// Number of forward simplification attempts (successful + unsuccessful)
    pub forward_attempts: usize,
    /// Total time spent on all forward simplification attempts
    pub forward_attempt_time: Duration,
    /// Number of backward simplification attempts (successful + unsuccessful)
    pub backward_attempts: usize,
    /// Total time spent on all backward simplification attempts
    pub backward_attempt_time: Duration,
}

impl Serialize for SimplificationStats {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("SimplificationStats", 8)?;
        s.serialize_field("forward_count", &self.forward_count)?;
        s.serialize_field("forward_time", &secs(&self.forward_time))?;
        s.serialize_field("backward_count", &self.backward_count)?;
        s.serialize_field("backward_time", &secs(&self.backward_time))?;
        s.serialize_field("forward_attempts", &self.forward_attempts)?;
        s.serialize_field("forward_attempt_time", &secs(&self.forward_attempt_time))?;
        s.serialize_field("backward_attempts", &self.backward_attempts)?;
        s.serialize_field("backward_attempt_time", &secs(&self.backward_attempt_time))?;
        s.end()
    }
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

    // Aggregate counters
    pub iterations: usize,
    pub clauses_generated: usize,
    pub clauses_added: usize,
    pub max_unprocessed_size: usize,
    pub max_processed_size: usize,

    // Dynamic rule stats
    pub generating_rules: HashMap<String, RuleStats>,
    pub simplification_rules: HashMap<String, SimplificationStats>,

    // Selector stats (filled post-saturation)
    pub selector_name: String,
    pub selector_cache_hits: usize,
    pub selector_cache_misses: usize,
    pub selector_embed_time: Duration,
    pub selector_score_time: Duration,
}

impl SaturationProfile {
    /// Record statistics for a generating inference rule.
    pub fn record_generating_rule(&mut self, name: &str, count: usize, time: Duration) {
        let stats = self.generating_rules.entry(name.to_string()).or_default();
        stats.count += count;
        stats.time += time;
    }

    /// Record forward simplification statistics.
    pub fn record_simplification_forward(&mut self, name: &str, count: usize, time: Duration) {
        let stats = self.simplification_rules.entry(name.to_string()).or_default();
        stats.forward_count += count;
        stats.forward_time += time;
    }

    /// Record backward simplification statistics.
    pub fn record_simplification_backward(&mut self, name: &str, count: usize, time: Duration) {
        let stats = self.simplification_rules.entry(name.to_string()).or_default();
        stats.backward_count += count;
        stats.backward_time += time;
    }

    /// Record a forward simplification attempt (successful or not).
    pub fn record_simplification_forward_attempt(&mut self, name: &str, success: bool, time: Duration) {
        let stats = self.simplification_rules.entry(name.to_string()).or_default();
        stats.forward_attempts += 1;
        stats.forward_attempt_time += time;
        if success {
            stats.forward_count += 1;
            stats.forward_time += time;
        }
    }

    /// Record a backward simplification attempt (successful or not).
    pub fn record_simplification_backward_attempt(&mut self, name: &str, count: usize, time: Duration) {
        let stats = self.simplification_rules.entry(name.to_string()).or_default();
        stats.backward_attempts += 1;
        stats.backward_attempt_time += time;
        if count > 0 {
            stats.backward_count += count;
            stats.backward_time += time;
        }
    }
}

impl Serialize for SaturationProfile {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("SaturationProfile", 15)?;

        // Top-level phase timings
        s.serialize_field("total_time", &secs(&self.total_time))?;
        s.serialize_field("forward_simplify_time", &secs(&self.forward_simplify_time))?;
        s.serialize_field("select_given_time", &secs(&self.select_given_time))?;
        s.serialize_field("generate_inferences_time", &secs(&self.generate_inferences_time))?;
        s.serialize_field("add_inferences_time", &secs(&self.add_inferences_time))?;

        // Aggregate counters
        s.serialize_field("iterations", &self.iterations)?;
        s.serialize_field("clauses_generated", &self.clauses_generated)?;
        s.serialize_field("clauses_added", &self.clauses_added)?;
        s.serialize_field("max_unprocessed_size", &self.max_unprocessed_size)?;
        s.serialize_field("max_processed_size", &self.max_processed_size)?;

        // Dynamic rule stats
        s.serialize_field("generating_rules", &self.generating_rules)?;
        s.serialize_field("simplification_rules", &self.simplification_rules)?;

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
        assert!(json.contains("\"generating_rules\":{}"));
        assert!(json.contains("\"simplification_rules\":{}"));
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
        profile.record_generating_rule("Resolution", 10, Duration::from_millis(100));

        let json = serde_json::to_string(&profile).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["total_time"], 2.0);
        assert_eq!(value["generating_rules"]["Resolution"]["count"], 10);
        assert_eq!(value["generating_rules"]["Resolution"]["time"], 0.1);
    }

    #[test]
    fn test_record_generating_rule() {
        let mut profile = SaturationProfile::default();
        profile.record_generating_rule("Resolution", 5, Duration::from_millis(50));
        profile.record_generating_rule("Resolution", 3, Duration::from_millis(30));
        profile.record_generating_rule("Superposition", 2, Duration::from_millis(20));

        assert_eq!(profile.generating_rules["Resolution"].count, 8);
        assert_eq!(profile.generating_rules["Resolution"].time, Duration::from_millis(80));
        assert_eq!(profile.generating_rules["Superposition"].count, 2);
    }

    #[test]
    fn test_record_simplification() {
        let mut profile = SaturationProfile::default();
        profile.record_simplification_forward("Subsumption", 1, Duration::from_millis(10));
        profile.record_simplification_backward("Subsumption", 3, Duration::from_millis(30));
        profile.record_simplification_forward("Demodulation", 2, Duration::from_millis(20));

        assert_eq!(profile.simplification_rules["Subsumption"].forward_count, 1);
        assert_eq!(profile.simplification_rules["Subsumption"].backward_count, 3);
        assert_eq!(profile.simplification_rules["Demodulation"].forward_count, 2);
        assert_eq!(profile.simplification_rules["Demodulation"].backward_count, 0);
    }

    #[test]
    fn test_simplification_stats_serializes() {
        let mut profile = SaturationProfile::default();
        profile.record_simplification_forward("Subsumption", 5, Duration::from_millis(100));
        profile.record_simplification_backward("Subsumption", 3, Duration::from_millis(50));

        let json = serde_json::to_string(&profile).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        let subsumption = &value["simplification_rules"]["Subsumption"];
        assert_eq!(subsumption["forward_count"], 5);
        assert_eq!(subsumption["forward_time"], 0.1);
        assert_eq!(subsumption["backward_count"], 3);
        assert_eq!(subsumption["backward_time"], 0.05);
    }
}
