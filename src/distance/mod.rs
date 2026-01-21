//! Distance metric calculations

use crate::error::{Error, Result};
use crate::vector::{Vector, VectorType};

pub mod scalar;
// Note: SIMD optimization is provided by the simsimd crate (used in scalar.rs)
// which automatically detects CPU features (AVX512/AVX2/SSE/NEON) at runtime

/// Distance metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// L2 / Euclidean distance: sqrt(sum((a - b)^2))
    L2,
    /// L1 / Manhattan distance: sum(|a - b|)
    L1,
    /// Cosine distance: 1 - (a · b) / (||a|| * ||b||)
    Cosine,
    /// Hamming distance: count of differing bits
    Hamming,
}

impl DistanceMetric {
    /// Parse distance metric from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(DistanceMetric::L2),
            "l1" | "manhattan" => Ok(DistanceMetric::L1),
            "cosine" => Ok(DistanceMetric::Cosine),
            "hamming" => Ok(DistanceMetric::Hamming),
            _ => Err(Error::InvalidDistanceMetric(s.to_string())),
        }
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            DistanceMetric::L2 => "l2",
            DistanceMetric::L1 => "l1",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Hamming => "hamming",
        }
    }
}

/// Calculate distance between two vectors
pub fn distance(a: &Vector, b: &Vector, metric: DistanceMetric) -> Result<f32> {
    if a.dimensions() != b.dimensions() {
        return Err(Error::DimensionMismatch {
            expected: a.dimensions(),
            actual: b.dimensions(),
        });
    }

    if a.vec_type() != b.vec_type() {
        return Err(Error::InvalidVectorType(
            "Vector types must match for distance calculation".to_string(),
        ));
    }

    match (a.vec_type(), metric) {
        (VectorType::Float32, DistanceMetric::L2) => distance_l2_f32(a, b),
        (VectorType::Float32, DistanceMetric::L1) => distance_l1_f32(a, b),
        (VectorType::Float32, DistanceMetric::Cosine) => distance_cosine_f32(a, b),
        (VectorType::Int8, DistanceMetric::L2) => distance_l2_i8(a, b),
        (VectorType::Int8, DistanceMetric::L1) => distance_l1_i8(a, b),
        (VectorType::Int8, DistanceMetric::Cosine) => distance_cosine_i8(a, b),
        (VectorType::Bit, DistanceMetric::Hamming) => distance_hamming(a, b),
        _ => Err(Error::InvalidDistanceMetric(format!(
            "Distance metric {:?} not supported for vector type {:?}",
            metric,
            a.vec_type()
        ))),
    }
}

/// L2 distance for Float32 vectors
/// Uses simsimd with automatic SIMD detection (AVX512/AVX2/SSE/NEON)
pub fn distance_l2_f32(a: &Vector, b: &Vector) -> Result<f32> {
    scalar::distance_l2_f32_scalar(a, b)
}

/// L1 distance for Float32 vectors
/// Pure Rust implementation (simsimd doesn't provide L1)
pub fn distance_l1_f32(a: &Vector, b: &Vector) -> Result<f32> {
    scalar::distance_l1_f32_scalar(a, b)
}

/// Cosine distance for Float32 vectors
/// Uses simsimd with automatic SIMD detection (AVX512/AVX2/SSE/NEON)
pub fn distance_cosine_f32(a: &Vector, b: &Vector) -> Result<f32> {
    scalar::distance_cosine_f32_scalar(a, b)
}

/// L2 distance for Int8 vectors
/// Uses simsimd with automatic SIMD detection (AVX512/AVX2/SSE/NEON)
pub fn distance_l2_i8(a: &Vector, b: &Vector) -> Result<f32> {
    scalar::distance_l2_i8_scalar(a, b)
}

/// L1 distance for Int8 vectors
/// Pure Rust implementation (simsimd doesn't provide L1 for int8)
pub fn distance_l1_i8(a: &Vector, b: &Vector) -> Result<f32> {
    scalar::distance_l1_i8_scalar(a, b)
}

/// Cosine distance for Int8 vectors
/// Uses simsimd with automatic SIMD detection (AVX512/AVX2/SSE/NEON)
pub fn distance_cosine_i8(a: &Vector, b: &Vector) -> Result<f32> {
    scalar::distance_cosine_i8_scalar(a, b)
}

/// Hamming distance for binary vectors
/// Uses simsimd with automatic SIMD detection
pub fn distance_hamming(a: &Vector, b: &Vector) -> Result<f32> {
    scalar::distance_hamming_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_from_str() {
        assert_eq!(DistanceMetric::from_str("l2").unwrap(), DistanceMetric::L2);
        assert_eq!(
            DistanceMetric::from_str("euclidean").unwrap(),
            DistanceMetric::L2
        );
        assert_eq!(
            DistanceMetric::from_str("cosine").unwrap(),
            DistanceMetric::Cosine
        );
        assert!(DistanceMetric::from_str("invalid").is_err());
    }

    #[test]
    fn test_distance_dimension_mismatch() {
        let vec1 = Vector::from_f32(&[1.0, 2.0]);
        let vec2 = Vector::from_f32(&[1.0, 2.0, 3.0]);

        let result = distance(&vec1, &vec2, DistanceMetric::L2);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::DimensionMismatch { .. })));
    }

    #[test]
    fn test_distance_l2_works() {
        let vec1 = Vector::from_f32(&[1.0, 2.0, 3.0]);
        let vec2 = Vector::from_f32(&[4.0, 5.0, 6.0]);

        let result = distance(&vec1, &vec2, DistanceMetric::L2);
        assert!(result.is_ok());

        // Distance should be sqrt(27) ≈ 5.196
        let dist = result.unwrap();
        assert!((dist - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_distance_cosine_works() {
        let vec1 = Vector::from_f32(&[1.0, 0.0, 0.0]);
        let vec2 = Vector::from_f32(&[0.0, 1.0, 0.0]);

        let result = distance(&vec1, &vec2, DistanceMetric::Cosine);
        assert!(result.is_ok());

        // Orthogonal vectors have cosine distance of 1
        let dist = result.unwrap();
        assert!((dist - 1.0).abs() < 0.01);
    }
}
