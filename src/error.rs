//! Error types for sqlite-vec-hnsw

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Invalid vector format: {0}")]
    InvalidVectorFormat(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid vector type: {0}")]
    InvalidVectorType(String),

    #[error("Invalid distance metric: {0}")]
    InvalidDistanceMetric(String),

    #[error("HNSW error: {0}")]
    Hnsw(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("JSON parse error: {0}")]
    JsonParse(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
