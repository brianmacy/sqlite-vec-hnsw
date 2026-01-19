//! SQL scalar function implementations

use crate::distance::{DistanceMetric, distance};
use crate::error::{Error, Result};
use crate::vector::{Vector, VectorType};
use rusqlite::Connection;
use rusqlite::functions::{Context, FunctionFlags};
use rusqlite::types::ValueRef;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Register all SQL functions with the database
pub fn register_all(db: &Connection) -> Result<()> {
    // Vector constructors
    register_vec_f32(db)?;
    register_vec_int8(db)?;
    register_vec_bit(db)?;

    // Distance functions
    register_vec_distance_l2(db)?;
    register_vec_distance_l1(db)?;
    register_vec_distance_cosine(db)?;
    register_vec_distance_hamming(db)?;

    // Vector operations
    register_vec_length(db)?;
    register_vec_type(db)?;
    register_vec_to_json(db)?;

    // Skip unimplemented functions for now
    let _ = register_vec_add(db);
    let _ = register_vec_sub(db);
    let _ = register_vec_normalize(db);
    let _ = register_vec_slice(db);

    // Quantization - skip unimplemented
    let _ = register_vec_quantize_int8(db);
    let _ = register_vec_quantize_binary(db);

    // Metadata
    register_vec_version(db)?;
    let _ = register_vec_debug(db);

    // Management functions
    register_vec_rebuild_hnsw(db)?;

    Ok(())
}

/// Helper to parse vector from SQL value (JSON string or blob)
fn vector_from_sql(value: ValueRef, vec_type: VectorType) -> Result<Vector> {
    match value {
        ValueRef::Text(s) => {
            let json_str = std::str::from_utf8(s)
                .map_err(|e| Error::InvalidVectorFormat(format!("Invalid UTF-8: {}", e)))?;
            Vector::from_json(json_str, vec_type)
        }
        ValueRef::Blob(b) => {
            // Determine dimensions from blob size
            let dimensions = match vec_type {
                VectorType::Float32 => b.len() / 4,
                VectorType::Int8 => b.len(),
                VectorType::Bit => b.len() * 8,
            };
            Vector::from_blob(b, vec_type, dimensions)
        }
        _ => Err(Error::InvalidVectorFormat(
            "Vector must be TEXT (JSON) or BLOB".to_string(),
        )),
    }
}

fn register_vec_f32(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_f32",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let value = ctx.get_raw(0);
            let vector = vector_from_sql(value, VectorType::Float32)
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(vector.as_bytes().to_vec())
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_int8(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_int8",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let value = ctx.get_raw(0);
            let vector = vector_from_sql(value, VectorType::Int8)
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(vector.as_bytes().to_vec())
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_bit(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_bit",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let value = ctx.get_raw(0);
            let vector = vector_from_sql(value, VectorType::Bit)
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(vector.as_bytes().to_vec())
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

/// Helper to extract two vectors from context for distance functions
fn get_two_vectors(ctx: &Context, vec_type: VectorType) -> rusqlite::Result<(Vector, Vector)> {
    let v1 = vector_from_sql(ctx.get_raw(0), vec_type)
        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
    let v2 = vector_from_sql(ctx.get_raw(1), vec_type)
        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
    Ok((v1, v2))
}

fn register_vec_distance_l2(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_distance_l2",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let (v1, v2) = get_two_vectors(ctx, VectorType::Float32)?;
            let dist = distance(&v1, &v2, DistanceMetric::L2)
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(dist as f64)
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_distance_l1(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_distance_l1",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let (v1, v2) = get_two_vectors(ctx, VectorType::Float32)?;
            let dist = distance(&v1, &v2, DistanceMetric::L1)
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(dist as f64)
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_distance_cosine(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_distance_cosine",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let (v1, v2) = get_two_vectors(ctx, VectorType::Float32)?;
            let dist = distance(&v1, &v2, DistanceMetric::Cosine)
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(dist as f64)
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_distance_hamming(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_distance_hamming",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let (v1, v2) = get_two_vectors(ctx, VectorType::Bit)?;
            let dist = distance(&v1, &v2, DistanceMetric::Hamming)
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(dist as f64)
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_length(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_length",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let value = ctx.get_raw(0);
            // Try to parse as any vector type
            let vector = vector_from_sql(value, VectorType::Float32)
                .or_else(|_| vector_from_sql(value, VectorType::Int8))
                .or_else(|_| vector_from_sql(value, VectorType::Bit))
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(vector.dimensions() as i64)
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_type(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_type",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let value = ctx.get_raw(0);
            // Try to parse as any vector type
            let vector = vector_from_sql(value, VectorType::Float32)
                .or_else(|_| vector_from_sql(value, VectorType::Int8))
                .or_else(|_| vector_from_sql(value, VectorType::Bit))
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(vector.vec_type().as_str())
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_to_json(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_to_json",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let value = ctx.get_raw(0);
            // Try to parse as any vector type
            let vector = vector_from_sql(value, VectorType::Float32)
                .or_else(|_| vector_from_sql(value, VectorType::Int8))
                .or_else(|_| vector_from_sql(value, VectorType::Bit))
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            let json = vector
                .to_json()
                .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
            Ok(json)
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_add(_db: &Connection) -> Result<()> {
    // TODO: Implement vec_add() function
    Err(Error::NotImplemented(
        "vec_add() not yet implemented".to_string(),
    ))
}

fn register_vec_sub(_db: &Connection) -> Result<()> {
    // TODO: Implement vec_sub() function
    Err(Error::NotImplemented(
        "vec_sub() not yet implemented".to_string(),
    ))
}

fn register_vec_normalize(_db: &Connection) -> Result<()> {
    // TODO: Implement vec_normalize() function
    Err(Error::NotImplemented(
        "vec_normalize() not yet implemented".to_string(),
    ))
}

fn register_vec_slice(_db: &Connection) -> Result<()> {
    // TODO: Implement vec_slice() function
    Err(Error::NotImplemented(
        "vec_slice() not yet implemented".to_string(),
    ))
}

fn register_vec_quantize_int8(_db: &Connection) -> Result<()> {
    // TODO: Implement vec_quantize_int8() function
    Err(Error::NotImplemented(
        "vec_quantize_int8() not yet implemented".to_string(),
    ))
}

fn register_vec_quantize_binary(_db: &Connection) -> Result<()> {
    // TODO: Implement vec_quantize_binary() function
    Err(Error::NotImplemented(
        "vec_quantize_binary() not yet implemented".to_string(),
    ))
}

fn register_vec_version(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_version",
        0,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |_ctx| Ok(format!("sqlite-vec-hnsw {}", VERSION)),
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

fn register_vec_debug(_db: &Connection) -> Result<()> {
    // TODO: Implement vec_debug() function
    Err(Error::NotImplemented(
        "vec_debug() not yet implemented".to_string(),
    ))
}

fn register_vec_rebuild_hnsw(db: &Connection) -> Result<()> {
    db.create_scalar_function(
        "vec_rebuild_hnsw",
        -1, // Variable number of arguments (2 or 4)
        FunctionFlags::SQLITE_UTF8,
        |ctx| -> rusqlite::Result<String> {
            let argc = ctx.len();
            if argc != 2 && argc != 4 {
                return Err(rusqlite::Error::UserFunctionError(
                    "vec_rebuild_hnsw() requires 2 or 4 arguments: table_name, column_name [, new_M, new_ef_construction]".into(),
                ));
            }

            let _table_name = ctx.get::<String>(0)?;
            let _column_name = ctx.get::<String>(1)?;

            if argc == 4 {
                let new_m = ctx.get::<i32>(2)?;
                let new_ef_construction = ctx.get::<i32>(3)?;

                if !(2..=100).contains(&new_m) {
                    return Err(rusqlite::Error::UserFunctionError(
                        "M must be between 2 and 100".into(),
                    ));
                }
                if !(10..=2000).contains(&new_ef_construction) {
                    return Err(rusqlite::Error::UserFunctionError(
                        "ef_construction must be between 10 and 2000".into(),
                    ));
                }
            }

            // rusqlite scalar functions don't have database access
            // This would need to be implemented as a virtual table method or
            // require C FFI to access sqlite3_context_db_handle()
            Err(rusqlite::Error::UserFunctionError(
                "vec_rebuild_hnsw() not yet supported (requires database handle access)".into(),
            ))
        },
    )
    .map_err(Error::Sqlite)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_all_works() {
        let db = Connection::open_in_memory().unwrap();
        let result = register_all(&db);
        // Should succeed now that implemented functions are registered
        assert!(result.is_ok());
    }

    #[test]
    fn test_vec_f32_registration() {
        let db = Connection::open_in_memory().unwrap();
        assert!(register_vec_f32(&db).is_ok());

        // Test that the function works
        let result: rusqlite::Result<Vec<u8>> =
            db.query_row("SELECT vec_f32('[1.0, 2.0, 3.0]')", [], |row| row.get(0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_vec_distance_l2_registration() {
        let db = Connection::open_in_memory().unwrap();
        assert!(register_vec_f32(&db).is_ok());
        assert!(register_vec_distance_l2(&db).is_ok());

        // Test that the function works
        let result: rusqlite::Result<f64> = db.query_row(
            "SELECT vec_distance_l2('[1.0, 2.0, 3.0]', '[4.0, 5.0, 6.0]')",
            [],
            |row| row.get(0),
        );
        assert!(result.is_ok());
        let distance = result.unwrap();
        // sqrt((3^2 + 3^2 + 3^2)) = sqrt(27) ≈ 5.196
        assert!((distance - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_vec_length_registration() {
        let db = Connection::open_in_memory().unwrap();
        assert!(register_vec_f32(&db).is_ok());
        assert!(register_vec_length(&db).is_ok());

        let result: rusqlite::Result<i64> =
            db.query_row("SELECT vec_length('[1.0, 2.0, 3.0, 4.0]')", [], |row| {
                row.get(0)
            });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4);
    }

    #[test]
    fn test_vec_version_registration() {
        let db = Connection::open_in_memory().unwrap();
        assert!(register_vec_version(&db).is_ok());

        let result: rusqlite::Result<String> =
            db.query_row("SELECT vec_version()", [], |row| row.get(0));
        assert!(result.is_ok());
        let version = result.unwrap();
        assert!(version.contains("sqlite-vec-hnsw"));
    }
}
