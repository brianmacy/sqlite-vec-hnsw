//! Vector types and operations

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Vector element types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorType {
    /// 32-bit floating point (4 bytes per element)
    Float32,
    /// 8-bit signed integer (1 byte per element)
    Int8,
    /// Binary/bit vectors (1 bit per element)
    Bit,
}

impl VectorType {
    /// Get the size in bytes for this type (per element)
    pub fn bytes_per_element(&self) -> usize {
        match self {
            VectorType::Float32 => 4,
            VectorType::Int8 => 1,
            VectorType::Bit => 0, // Handled specially (8 elements per byte)
        }
    }

    /// Parse type from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "float32" | "float" => Ok(VectorType::Float32),
            "int8" => Ok(VectorType::Int8),
            "bit" | "binary" => Ok(VectorType::Bit),
            _ => Err(Error::InvalidVectorType(s.to_string())),
        }
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            VectorType::Float32 => "float32",
            VectorType::Int8 => "int8",
            VectorType::Bit => "bit",
        }
    }
}

/// Vector data container
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    vec_type: VectorType,
    dimensions: usize,
    data: Vec<u8>,
}

impl Vector {
    /// Create a new Float32 vector from slice
    pub fn from_f32(values: &[f32]) -> Self {
        let mut data = Vec::with_capacity(values.len() * 4);
        for &v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Vector {
            vec_type: VectorType::Float32,
            dimensions: values.len(),
            data,
        }
    }

    /// Create a new Int8 vector from slice
    pub fn from_i8(values: &[i8]) -> Self {
        Vector {
            vec_type: VectorType::Int8,
            dimensions: values.len(),
            data: values.iter().map(|&v| v as u8).collect(),
        }
    }

    /// Create vector from JSON string
    pub fn from_json(json: &str, vec_type: VectorType) -> Result<Self> {
        let values: Vec<f64> = serde_json::from_str(json)?;

        match vec_type {
            VectorType::Float32 => {
                let f32_values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                Ok(Self::from_f32(&f32_values))
            }
            VectorType::Int8 => {
                let i8_values: Vec<i8> = values.iter().map(|&v| v as i8).collect();
                Ok(Self::from_i8(&i8_values))
            }
            VectorType::Bit => Err(Error::NotImplemented(
                "Binary vector from JSON not yet implemented".to_string(),
            )),
        }
    }

    /// Create vector from binary blob
    pub fn from_blob(blob: &[u8], vec_type: VectorType, dimensions: usize) -> Result<Self> {
        // TODO: Validate blob size matches expected size for type and dimensions
        Ok(Vector {
            vec_type,
            dimensions,
            data: blob.to_vec(),
        })
    }

    /// Get vector type
    pub fn vec_type(&self) -> VectorType {
        self.vec_type
    }

    /// Get number of dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get raw data as slice
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Convert to f32 slice (for Float32 vectors)
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.vec_type != VectorType::Float32 {
            return Err(Error::InvalidVectorType(
                "Vector is not Float32 type".to_string(),
            ));
        }

        let mut result = Vec::with_capacity(self.dimensions);
        for chunk in self.data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            result.push(f32::from_le_bytes(bytes));
        }
        Ok(result)
    }

    /// Convert to i8 slice (for Int8 vectors)
    pub fn as_i8(&self) -> Result<Vec<i8>> {
        if self.vec_type != VectorType::Int8 {
            return Err(Error::InvalidVectorType(
                "Vector is not Int8 type".to_string(),
            ));
        }

        Ok(self.data.iter().map(|&b| b as i8).collect())
    }

    /// Convert vector to JSON string
    pub fn to_json(&self) -> Result<String> {
        match self.vec_type {
            VectorType::Float32 => {
                let values = self.as_f32()?;
                Ok(serde_json::to_string(&values)?)
            }
            VectorType::Int8 => {
                let values = self.as_i8()?;
                Ok(serde_json::to_string(&values)?)
            }
            VectorType::Bit => Err(Error::NotImplemented(
                "Binary vector to JSON not yet implemented".to_string(),
            )),
        }
    }

    /// Add two vectors element-wise
    pub fn add(&self, other: &Vector) -> Result<Vector> {
        if self.dimensions != other.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.dimensions,
                actual: other.dimensions,
            });
        }

        if self.vec_type != other.vec_type {
            return Err(Error::InvalidVectorType(
                "Vector types must match for addition".to_string(),
            ));
        }

        match self.vec_type {
            VectorType::Float32 => {
                let a = self.as_f32()?;
                let b = other.as_f32()?;
                let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                Ok(Vector::from_f32(&result))
            }
            VectorType::Int8 => {
                let a = self.as_i8()?;
                let b = other.as_i8()?;
                let result: Vec<i8> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.saturating_add(*y))
                    .collect();
                Ok(Vector::from_i8(&result))
            }
            VectorType::Bit => Err(Error::InvalidVectorType(
                "Cannot add binary vectors".to_string(),
            )),
        }
    }

    /// Subtract two vectors element-wise
    pub fn sub(&self, other: &Vector) -> Result<Vector> {
        if self.dimensions != other.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.dimensions,
                actual: other.dimensions,
            });
        }

        if self.vec_type != other.vec_type {
            return Err(Error::InvalidVectorType(
                "Vector types must match for subtraction".to_string(),
            ));
        }

        match self.vec_type {
            VectorType::Float32 => {
                let a = self.as_f32()?;
                let b = other.as_f32()?;
                let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
                Ok(Vector::from_f32(&result))
            }
            VectorType::Int8 => {
                let a = self.as_i8()?;
                let b = other.as_i8()?;
                let result: Vec<i8> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.saturating_sub(*y))
                    .collect();
                Ok(Vector::from_i8(&result))
            }
            VectorType::Bit => Err(Error::InvalidVectorType(
                "Cannot subtract binary vectors".to_string(),
            )),
        }
    }

    /// Normalize vector to unit length (L2 norm)
    pub fn normalize(&self) -> Result<Vector> {
        match self.vec_type {
            VectorType::Float32 => {
                let vals = self.as_f32()?;
                let magnitude: f32 = vals.iter().map(|x| x * x).sum::<f32>().sqrt();

                if magnitude == 0.0 {
                    return Err(Error::InvalidParameter(
                        "Cannot normalize zero vector".to_string(),
                    ));
                }

                let result: Vec<f32> = vals.iter().map(|x| x / magnitude).collect();
                Ok(Vector::from_f32(&result))
            }
            VectorType::Int8 => Err(Error::InvalidVectorType(
                "Cannot normalize Int8 vectors (would lose precision)".to_string(),
            )),
            VectorType::Bit => Err(Error::InvalidVectorType(
                "Cannot normalize binary vectors".to_string(),
            )),
        }
    }

    /// Slice vector (extract sub-vector from start to end, exclusive)
    pub fn slice(&self, start: usize, end: usize) -> Result<Vector> {
        if start >= self.dimensions || end > self.dimensions || start >= end {
            return Err(Error::InvalidParameter(format!(
                "Invalid slice range: {}..{} for vector of length {}",
                start, end, self.dimensions
            )));
        }

        match self.vec_type {
            VectorType::Float32 => {
                let vals = self.as_f32()?;
                let sliced = vals[start..end].to_vec();
                Ok(Vector::from_f32(&sliced))
            }
            VectorType::Int8 => {
                let vals = self.as_i8()?;
                let sliced = vals[start..end].to_vec();
                Ok(Vector::from_i8(&sliced))
            }
            VectorType::Bit => {
                // For binary vectors, slice at byte boundaries
                if !start.is_multiple_of(8) || !end.is_multiple_of(8) {
                    return Err(Error::InvalidParameter(
                        "Binary vector slice must be at byte boundaries (multiples of 8)"
                            .to_string(),
                    ));
                }

                let start_byte = start / 8;
                let end_byte = end / 8;
                let sliced = self.data[start_byte..end_byte].to_vec();

                Ok(Vector {
                    data: sliced,
                    dimensions: end - start,
                    vec_type: VectorType::Bit,
                })
            }
        }
    }

    /// Quantize float32 vector to int8 (asymmetric quantization)
    /// Maps the range [min, max] to [-128, 127]
    pub fn quantize_int8(&self) -> Result<Vector> {
        if self.vec_type != VectorType::Float32 {
            return Err(Error::InvalidVectorType(
                "Can only quantize Float32 vectors".to_string(),
            ));
        }

        let vals = self.as_f32()?;

        // Find min and max values
        let min_val = vals.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Handle edge cases
        if min_val == max_val {
            // All values are the same
            return Ok(Vector::from_i8(&vec![0i8; vals.len()]));
        }

        // Scale to int8 range [-128, 127]
        let range = max_val - min_val;
        let quantized: Vec<i8> = vals
            .iter()
            .map(|&v| {
                let normalized = (v - min_val) / range; // [0, 1]
                let scaled = normalized * 255.0 - 128.0; // [-128, 127]
                scaled.round().clamp(-128.0, 127.0) as i8
            })
            .collect();

        Ok(Vector::from_i8(&quantized))
    }

    /// Quantize to binary vector (threshold at mean)
    /// Values above mean become 1, below become 0
    pub fn quantize_binary(&self) -> Result<Vector> {
        if self.vec_type != VectorType::Float32 {
            return Err(Error::InvalidVectorType(
                "Can only quantize Float32 vectors to binary".to_string(),
            ));
        }

        let vals = self.as_f32()?;

        // Calculate mean as threshold
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;

        // Convert to bits, pack into bytes
        let num_bytes = vals.len().div_ceil(8);
        let mut bytes = vec![0u8; num_bytes];

        for (i, &val) in vals.iter().enumerate() {
            if val >= mean {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(Vector {
            data: bytes,
            dimensions: vals.len(),
            vec_type: VectorType::Bit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_type_bytes_per_element() {
        assert_eq!(VectorType::Float32.bytes_per_element(), 4);
        assert_eq!(VectorType::Int8.bytes_per_element(), 1);
        assert_eq!(VectorType::Bit.bytes_per_element(), 0);
    }

    #[test]
    fn test_vector_type_from_str() {
        assert_eq!(
            VectorType::from_str("float32").unwrap(),
            VectorType::Float32
        );
        assert_eq!(VectorType::from_str("float").unwrap(), VectorType::Float32);
        assert_eq!(VectorType::from_str("int8").unwrap(), VectorType::Int8);
        assert_eq!(VectorType::from_str("bit").unwrap(), VectorType::Bit);
        assert!(VectorType::from_str("invalid").is_err());
    }

    #[test]
    fn test_vector_from_f32() {
        let values = vec![1.0, 2.0, 3.0];
        let vec = Vector::from_f32(&values);

        assert_eq!(vec.vec_type(), VectorType::Float32);
        assert_eq!(vec.dimensions(), 3);
        assert_eq!(vec.as_f32().unwrap(), values);
    }

    #[test]
    fn test_vector_from_i8() {
        let values = vec![-128, 0, 127];
        let vec = Vector::from_i8(&values);

        assert_eq!(vec.vec_type(), VectorType::Int8);
        assert_eq!(vec.dimensions(), 3);
        assert_eq!(vec.as_i8().unwrap(), values);
    }

    #[test]
    fn test_vector_from_json_float32() {
        let json = "[1.0, 2.0, 3.0]";
        let vec = Vector::from_json(json, VectorType::Float32).unwrap();

        assert_eq!(vec.dimensions(), 3);
        let values = vec.as_f32().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_from_json_int8() {
        let json = "[-128, 0, 127]";
        let vec = Vector::from_json(json, VectorType::Int8).unwrap();

        assert_eq!(vec.dimensions(), 3);
        let values = vec.as_i8().unwrap();
        assert_eq!(values, vec![-128, 0, 127]);
    }

    #[test]
    fn test_vector_to_json_float32() {
        let values = vec![1.0, 2.0, 3.0];
        let vec = Vector::from_f32(&values);
        let json = vec.to_json().unwrap();

        assert_eq!(json, "[1.0,2.0,3.0]");
    }

    #[test]
    fn test_vector_add_dimension_mismatch() {
        let vec1 = Vector::from_f32(&[1.0, 2.0]);
        let vec2 = Vector::from_f32(&[1.0, 2.0, 3.0]);

        let result = vec1.add(&vec2);
        assert!(result.is_err());
        match result {
            Err(Error::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_vector_add() {
        let vec1 = Vector::from_f32(&[1.0, 2.0, 3.0]);
        let vec2 = Vector::from_f32(&[4.0, 5.0, 6.0]);

        let result = vec1.add(&vec2).unwrap();
        let vals = result.as_f32().unwrap();
        assert_eq!(vals, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_sub() {
        let vec1 = Vector::from_f32(&[4.0, 5.0, 6.0]);
        let vec2 = Vector::from_f32(&[1.0, 2.0, 3.0]);

        let result = vec1.sub(&vec2).unwrap();
        let vals = result.as_f32().unwrap();
        assert_eq!(vals, &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_vector_normalize() {
        let vec = Vector::from_f32(&[3.0, 4.0]);
        let result = vec.normalize().unwrap();
        let vals = result.as_f32().unwrap();

        // 3-4-5 triangle: magnitude is 5
        assert!((vals[0] - 0.6).abs() < 0.0001);
        assert!((vals[1] - 0.8).abs() < 0.0001);

        // Verify unit length
        let magnitude: f32 = vals.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_vector_slice() {
        let vec = Vector::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = vec.slice(1, 4).unwrap();
        let vals = result.as_f32().unwrap();
        assert_eq!(vals, &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vector_slice_invalid_range() {
        let vec = Vector::from_f32(&[1.0, 2.0, 3.0]);
        assert!(vec.slice(0, 10).is_err());
        assert!(vec.slice(2, 1).is_err());
        assert!(vec.slice(3, 4).is_err());
    }

    #[test]
    fn test_vector_quantize_int8() {
        let vec = Vector::from_f32(&[0.0, 0.5, 1.0]);
        let result = vec.quantize_int8().unwrap();
        let vals = result.as_i8().unwrap();

        // Should map [0, 0.5, 1.0] to [-128, 0, 127] approximately
        assert_eq!(vals.len(), 3);
        assert!(vals[0] < vals[1]);
        assert!(vals[1] < vals[2]);
        assert_eq!(vals[0], -128);
        assert_eq!(vals[2], 127);
    }

    #[test]
    fn test_vector_quantize_binary() {
        let vec = Vector::from_f32(&[-1.0, -0.5, 0.5, 1.0]);
        let result = vec.quantize_binary().unwrap();

        // Mean is 0, so values below 0 become 0, above become 1
        assert_eq!(result.vec_type(), VectorType::Bit);
        assert_eq!(result.dimensions(), 4);
    }
}
