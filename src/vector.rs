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

        // TODO: Implement actual addition
        Err(Error::NotImplemented(
            "vec_add not yet implemented".to_string(),
        ))
    }

    /// Subtract two vectors element-wise
    pub fn sub(&self, other: &Vector) -> Result<Vector> {
        if self.dimensions != other.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.dimensions,
                actual: other.dimensions,
            });
        }

        // TODO: Implement actual subtraction
        Err(Error::NotImplemented(
            "vec_sub not yet implemented".to_string(),
        ))
    }

    /// Normalize vector to unit length
    pub fn normalize(&self) -> Result<Vector> {
        // TODO: Implement normalization
        Err(Error::NotImplemented(
            "vec_normalize not yet implemented".to_string(),
        ))
    }

    /// Slice vector
    pub fn slice(&self, start: usize, end: usize) -> Result<Vector> {
        if start >= self.dimensions || end > self.dimensions || start >= end {
            return Err(Error::InvalidParameter(format!(
                "Invalid slice range: {}..{} for vector of length {}",
                start, end, self.dimensions
            )));
        }

        // TODO: Implement slicing
        Err(Error::NotImplemented(
            "vec_slice not yet implemented".to_string(),
        ))
    }

    /// Quantize float32 vector to int8
    pub fn quantize_int8(&self) -> Result<Vector> {
        if self.vec_type != VectorType::Float32 {
            return Err(Error::InvalidVectorType(
                "Can only quantize Float32 vectors".to_string(),
            ));
        }

        // TODO: Implement quantization
        Err(Error::NotImplemented(
            "vec_quantize_int8 not yet implemented".to_string(),
        ))
    }

    /// Quantize to binary vector
    pub fn quantize_binary(&self) -> Result<Vector> {
        // TODO: Implement binary quantization
        Err(Error::NotImplemented(
            "vec_quantize_binary not yet implemented".to_string(),
        ))
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
    fn test_vector_add_not_implemented() {
        let vec1 = Vector::from_f32(&[1.0, 2.0, 3.0]);
        let vec2 = Vector::from_f32(&[4.0, 5.0, 6.0]);

        let result = vec1.add(&vec2);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::NotImplemented(_))));
    }

    #[test]
    fn test_vector_normalize_not_implemented() {
        let vec = Vector::from_f32(&[3.0, 4.0]);
        let result = vec.normalize();
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::NotImplemented(_))));
    }

    #[test]
    fn test_vector_slice_invalid_range() {
        let vec = Vector::from_f32(&[1.0, 2.0, 3.0]);
        assert!(vec.slice(0, 10).is_err());
        assert!(vec.slice(2, 1).is_err());
        assert!(vec.slice(3, 4).is_err());
    }

    #[test]
    fn test_vector_quantize_not_implemented() {
        let vec = Vector::from_f32(&[0.5, -0.3, 0.8]);
        let result = vec.quantize_int8();
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::NotImplemented(_))));
    }
}
