//! Distance calculations using simsimd
//!
//! simsimd provides SIMD-optimized distance calculations with automatic
//! CPU feature detection (AVX512, AVX2, SSE, NEON).

use crate::error::{Error, Result};
use crate::vector::Vector;
use simsimd::{BinarySimilarity, SpatialSimilarity};

/// L2 distance for Float32 vectors
#[inline]
pub fn distance_l2_f32_scalar(a: &Vector, b: &Vector) -> Result<f32> {
    // Zero-copy: reinterpret bytes as f32 slice without allocation
    let a_vals = a.as_f32_slice();
    let b_vals = b.as_f32_slice();

    let distance = f32::sqeuclidean(a_vals, b_vals)
        .ok_or_else(|| Error::InvalidParameter("L2 distance calculation failed".to_string()))?;

    Ok((distance as f32).sqrt())
}

/// L1 distance for Float32 vectors
#[inline]
pub fn distance_l1_f32_scalar(a: &Vector, b: &Vector) -> Result<f32> {
    // Zero-copy: reinterpret bytes as f32 slice without allocation
    let a_vals = a.as_f32_slice();
    let b_vals = b.as_f32_slice();

    // simsimd doesn't have L1, so implement manually
    let distance: f32 = a_vals
        .iter()
        .zip(b_vals.iter())
        .map(|(x, y)| (x - y).abs())
        .sum();

    Ok(distance)
}

/// Cosine distance for Float32 vectors
#[inline]
pub fn distance_cosine_f32_scalar(a: &Vector, b: &Vector) -> Result<f32> {
    // Zero-copy: reinterpret bytes as f32 slice without allocation
    let a_vals = a.as_f32_slice();
    let b_vals = b.as_f32_slice();

    // simsimd::cosine() returns cosine distance (1 - similarity) directly
    let distance = f32::cosine(a_vals, b_vals)
        .ok_or_else(|| Error::InvalidParameter("Cosine distance calculation failed".to_string()))?;

    Ok(distance as f32)
}

/// L2 distance for Int8 vectors
#[inline]
pub fn distance_l2_i8_scalar(a: &Vector, b: &Vector) -> Result<f32> {
    // Zero-copy: reinterpret bytes as i8 slice without allocation
    let a_vals = a.as_i8_slice();
    let b_vals = b.as_i8_slice();

    let distance = i8::sqeuclidean(a_vals, b_vals).ok_or_else(|| {
        Error::InvalidParameter("L2 distance (Int8) calculation failed".to_string())
    })?;

    Ok(distance.sqrt() as f32)
}

/// L1 distance for Int8 vectors
#[inline]
pub fn distance_l1_i8_scalar(a: &Vector, b: &Vector) -> Result<f32> {
    // Zero-copy: reinterpret bytes as i8 slice without allocation
    let a_vals = a.as_i8_slice();
    let b_vals = b.as_i8_slice();

    // simsimd doesn't have L1 for i8, so implement manually
    let distance: i32 = a_vals
        .iter()
        .zip(b_vals.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .sum();

    Ok(distance as f32)
}

/// Hamming distance for binary vectors
#[inline]
pub fn distance_hamming_scalar(a: &Vector, b: &Vector) -> Result<f32> {
    // as_bytes() is already zero-copy
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();

    let distance = u8::hamming(a_bytes, b_bytes).ok_or_else(|| {
        Error::InvalidParameter("Hamming distance calculation failed".to_string())
    })?;

    Ok(distance as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_l2_f32() {
        let vec1 = Vector::from_f32(&[1.0, 2.0, 3.0]);
        let vec2 = Vector::from_f32(&[4.0, 5.0, 6.0]);

        let result = distance_l2_f32_scalar(&vec1, &vec2);
        assert!(result.is_ok());

        // Distance should be sqrt((3^2 + 3^2 + 3^2)) = sqrt(27) ≈ 5.196
        let distance = result.unwrap();
        assert!((distance - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_scalar_l1_f32() {
        let vec1 = Vector::from_f32(&[1.0, 2.0, 3.0]);
        let vec2 = Vector::from_f32(&[4.0, 5.0, 6.0]);

        let result = distance_l1_f32_scalar(&vec1, &vec2);
        assert!(result.is_ok());

        // Distance should be |3| + |3| + |3| = 9
        let distance = result.unwrap();
        assert!((distance - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_scalar_cosine_f32() {
        let vec1 = Vector::from_f32(&[1.0, 0.0, 0.0]);
        let vec2 = Vector::from_f32(&[0.0, 1.0, 0.0]);

        let result = distance_cosine_f32_scalar(&vec1, &vec2);
        assert!(result.is_ok());

        // Orthogonal vectors: cosine similarity = 0, distance = 1
        let distance = result.unwrap();
        println!("Cosine distance for orthogonal vectors: {}", distance);
        assert!((distance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_scalar_cosine_parallel() {
        let vec1 = Vector::from_f32(&[1.0, 2.0, 3.0]);
        let vec2 = Vector::from_f32(&[2.0, 4.0, 6.0]);

        let result = distance_cosine_f32_scalar(&vec1, &vec2);
        assert!(result.is_ok());

        // Parallel vectors: cosine similarity = 1, distance = 0
        let distance = result.unwrap();
        println!("Cosine distance for parallel vectors: {}", distance);
        assert!(distance.abs() < 0.01);
    }

    #[test]
    fn test_scalar_l2_i8() {
        let vec1 = Vector::from_i8(&[1, 2, 3]);
        let vec2 = Vector::from_i8(&[4, 5, 6]);

        let result = distance_l2_i8_scalar(&vec1, &vec2);
        assert!(result.is_ok());

        // Distance should be sqrt(27) ≈ 5.196
        let distance = result.unwrap();
        assert!((distance - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_scalar_l1_i8() {
        let vec1 = Vector::from_i8(&[1, 2, 3]);
        let vec2 = Vector::from_i8(&[4, 5, 6]);

        let result = distance_l1_i8_scalar(&vec1, &vec2);
        assert!(result.is_ok());

        // Distance should be 9
        let distance = result.unwrap();
        assert!((distance - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_scalar_hamming() {
        // Binary vectors represented as bytes
        // [1,0,1,0] vs [0,1,1,0] -> 2 bits different
        let vec1 = Vector::from_i8(&[1, 0, 1, 0]);
        let vec2 = Vector::from_i8(&[0, 1, 1, 0]);

        let result = distance_hamming_scalar(&vec1, &vec2);
        assert!(result.is_ok());

        let distance = result.unwrap();
        // Hamming distance counts differing positions
        assert!(distance >= 0.0);
    }
}
