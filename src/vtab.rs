//! Virtual table implementation for vec0

use crate::error::{Error, Result};
use rusqlite::Connection;

/// Register the vec0 virtual table module
pub fn register_vec0_module(_db: &Connection) -> Result<()> {
    // TODO: Implement vec0 virtual table registration
    Err(Error::NotImplemented(
        "vec0 virtual table not yet implemented".to_string(),
    ))
}

/// Query plan types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryPlan {
    /// Full table scan
    FullScan,
    /// Point query by rowid
    Point,
    /// K-nearest neighbor search
    Knn,
}

impl QueryPlan {
    /// Convert to idxStr header character
    pub fn to_header_char(&self) -> char {
        match self {
            QueryPlan::FullScan => '1',
            QueryPlan::Point => '2',
            QueryPlan::Knn => '3',
        }
    }

    /// Parse from idxStr header character
    pub fn from_header_char(c: char) -> Result<Self> {
        match c {
            '1' => Ok(QueryPlan::FullScan),
            '2' => Ok(QueryPlan::Point),
            '3' => Ok(QueryPlan::Knn),
            _ => Err(Error::InvalidParameter(format!(
                "Invalid query plan header: '{}'",
                c
            ))),
        }
    }
}

/// idxStr block types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdxStrKind {
    /// KNN query vector: '{___'
    KnnMatch,
    /// KNN k limit: '}___'
    KnnK,
    /// Rowid IN filter: '[___'
    KnnRowidIn,
    /// Partition constraint: ']Xop_' (X=column, op=operator)
    PartitionConstraint,
    /// Metadata constraint: '&Xop_' (X=column, op=operator)
    MetadataConstraint,
    /// Point query rowid: '!___'
    PointId,
}

impl IdxStrKind {
    /// Convert to 4-character block (first character)
    pub fn to_char(&self) -> char {
        match self {
            IdxStrKind::KnnMatch => '{',
            IdxStrKind::KnnK => '}',
            IdxStrKind::KnnRowidIn => '[',
            IdxStrKind::PartitionConstraint => ']',
            IdxStrKind::MetadataConstraint => '&',
            IdxStrKind::PointId => '!',
        }
    }

    /// Parse from first character of 4-character block
    pub fn from_char(c: char) -> Result<Self> {
        match c {
            '{' => Ok(IdxStrKind::KnnMatch),
            '}' => Ok(IdxStrKind::KnnK),
            '[' => Ok(IdxStrKind::KnnRowidIn),
            ']' => Ok(IdxStrKind::PartitionConstraint),
            '&' => Ok(IdxStrKind::MetadataConstraint),
            '!' => Ok(IdxStrKind::PointId),
            _ => Err(Error::InvalidParameter(format!(
                "Invalid idxStr block kind: '{}'",
                c
            ))),
        }
    }
}

/// Parse idxStr into components
pub fn parse_idxstr(idxstr: &str) -> Result<(QueryPlan, Vec<IdxStrKind>)> {
    if idxstr.is_empty() {
        return Err(Error::InvalidParameter("Empty idxStr".to_string()));
    }

    let mut chars = idxstr.chars();
    let header = chars.next().unwrap();
    let plan = QueryPlan::from_header_char(header)?;

    let mut blocks = Vec::new();
    let remaining: String = chars.collect();

    // Parse 4-character blocks
    for chunk in remaining.as_bytes().chunks(4) {
        if chunk.len() == 4 {
            let block_kind = IdxStrKind::from_char(chunk[0] as char)?;
            blocks.push(block_kind);
        }
    }

    Ok((plan, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_vec0_not_implemented() {
        let db = Connection::open_in_memory().unwrap();
        let result = register_vec0_module(&db);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::NotImplemented(_))));
    }

    #[test]
    fn test_query_plan_conversions() {
        assert_eq!(QueryPlan::FullScan.to_header_char(), '1');
        assert_eq!(QueryPlan::Point.to_header_char(), '2');
        assert_eq!(QueryPlan::Knn.to_header_char(), '3');

        assert_eq!(
            QueryPlan::from_header_char('1').unwrap(),
            QueryPlan::FullScan
        );
        assert_eq!(QueryPlan::from_header_char('2').unwrap(), QueryPlan::Point);
        assert_eq!(QueryPlan::from_header_char('3').unwrap(), QueryPlan::Knn);

        assert!(QueryPlan::from_header_char('X').is_err());
    }

    #[test]
    fn test_idxstr_kind_conversions() {
        assert_eq!(IdxStrKind::KnnMatch.to_char(), '{');
        assert_eq!(IdxStrKind::KnnK.to_char(), '}');
        assert_eq!(IdxStrKind::PointId.to_char(), '!');

        assert_eq!(IdxStrKind::from_char('{').unwrap(), IdxStrKind::KnnMatch);
        assert_eq!(IdxStrKind::from_char('}').unwrap(), IdxStrKind::KnnK);
        assert_eq!(IdxStrKind::from_char('!').unwrap(), IdxStrKind::PointId);

        assert!(IdxStrKind::from_char('X').is_err());
    }

    #[test]
    fn test_parse_idxstr_fullscan() {
        let (plan, blocks) = parse_idxstr("1").unwrap();
        assert_eq!(plan, QueryPlan::FullScan);
        assert_eq!(blocks.len(), 0);
    }

    #[test]
    fn test_parse_idxstr_knn() {
        // KNN query with match and k: "3{___}___"
        let (plan, blocks) = parse_idxstr("3{___}___").unwrap();
        assert_eq!(plan, QueryPlan::Knn);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], IdxStrKind::KnnMatch);
        assert_eq!(blocks[1], IdxStrKind::KnnK);
    }

    #[test]
    fn test_parse_idxstr_point() {
        // Point query: "2!___"
        let (plan, blocks) = parse_idxstr("2!___").unwrap();
        assert_eq!(plan, QueryPlan::Point);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], IdxStrKind::PointId);
    }

    #[test]
    fn test_parse_idxstr_empty() {
        let result = parse_idxstr("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_idxstr_invalid_header() {
        let result = parse_idxstr("X");
        assert!(result.is_err());
    }
}
