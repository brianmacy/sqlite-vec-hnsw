//! Virtual table implementation for vec0

use crate::error::{Error, Result};
use crate::vector::VectorType;
use rusqlite::Connection;
use rusqlite::vtab::{
    Context, CreateVTab, IndexInfo, UpdateVTab, VTab, VTabConnection, VTabCursor, Values,
    sqlite3_vtab, sqlite3_vtab_cursor,
};
use std::marker::PhantomData;
use std::os::raw::c_int;

/// Register the vec0 virtual table module
pub fn register_vec0_module(db: &Connection) -> Result<()> {
    let module = rusqlite::vtab::eponymous_only_module::<Vec0Tab>();
    db.create_module("vec0", module, None)
        .map_err(Error::Sqlite)?;
    Ok(())
}

/// Column definition for vec0 table
#[derive(Debug, Clone)]
struct ColumnDef {
    name: String,
    col_type: ColumnType,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ColumnType {
    Vector {
        vec_type: VectorType,
        dimensions: usize,
    },
    PartitionKey,
    Auxiliary,
    Metadata,
}

/// vec0 virtual table structure
#[repr(C)]
pub struct Vec0Tab {
    base: sqlite3_vtab,
    columns: Vec<ColumnDef>,
    rows: Vec<Row>,
}

#[derive(Debug, Clone)]
struct Row {
    rowid: i64,
    data: Vec<Option<Vec<u8>>>,
}

impl Vec0Tab {
    fn parse_create_args(args: &[&str]) -> Result<Vec<ColumnDef>> {
        let mut columns = Vec::new();

        // Skip module name and table name (first two args)
        for arg in args.iter().skip(2) {
            let arg = arg.trim();
            if arg.is_empty() {
                continue;
            }

            // Parse column definition: "column_name type[dimensions]"
            let parts: Vec<&str> = arg.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let name = parts[0].to_string();

            if parts.len() > 1 {
                let type_spec = parts[1];

                // Check for vector type with dimensions: float[768]
                if let Some(bracket_pos) = type_spec.find('[') {
                    let vec_type_str = &type_spec[..bracket_pos];
                    let dims_str = &type_spec[bracket_pos + 1..type_spec.len() - 1];

                    let vec_type = VectorType::from_str(vec_type_str)?;
                    let dimensions: usize = dims_str.parse().map_err(|_| {
                        Error::InvalidParameter(format!("Invalid dimensions: {}", dims_str))
                    })?;

                    columns.push(ColumnDef {
                        name,
                        col_type: ColumnType::Vector {
                            vec_type,
                            dimensions,
                        },
                    });
                } else {
                    // For now, treat other types as metadata
                    columns.push(ColumnDef {
                        name,
                        col_type: ColumnType::Metadata,
                    });
                }
            } else {
                // No type specified, treat as metadata
                columns.push(ColumnDef {
                    name,
                    col_type: ColumnType::Metadata,
                });
            }
        }

        Ok(columns)
    }
}

unsafe impl<'vtab> VTab<'vtab> for Vec0Tab {
    type Aux = ();
    type Cursor = Vec0TabCursor<'vtab>;

    fn connect(
        _db: &mut VTabConnection,
        _aux: Option<&Self::Aux>,
        args: &[&[u8]],
    ) -> rusqlite::Result<(String, Self)> {
        let args_str: Vec<&str> = args
            .iter()
            .map(|arg| std::str::from_utf8(arg).unwrap_or(""))
            .collect();

        let columns = Self::parse_create_args(&args_str)
            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;

        // Build CREATE TABLE statement for SQLite
        let mut sql = String::from("CREATE TABLE x(");
        for (i, col) in columns.iter().enumerate() {
            if i > 0 {
                sql.push_str(", ");
            }
            sql.push_str(&col.name);
            match &col.col_type {
                ColumnType::Vector { .. } => sql.push_str(" BLOB"),
                ColumnType::Metadata => sql.push_str(" TEXT"),
                _ => sql.push_str(" TEXT"),
            }
        }
        sql.push(')');

        Ok((
            sql,
            Vec0Tab {
                base: sqlite3_vtab::default(),
                columns,
                rows: Vec::new(),
            },
        ))
    }

    fn best_index(&self, info: &mut IndexInfo) -> rusqlite::Result<()> {
        // For now, just do a full scan
        // TODO: Implement proper query planning for KNN queries
        info.set_estimated_cost(1000000.0);
        info.set_estimated_rows(1000000);
        info.set_idx_str("1"); // FullScan query plan
        Ok(())
    }

    fn open(&mut self) -> rusqlite::Result<Vec0TabCursor<'vtab>> {
        Ok(Vec0TabCursor::new(self))
    }
}

impl<'vtab> CreateVTab<'vtab> for Vec0Tab {
    const KIND: rusqlite::vtab::VTabKind = rusqlite::vtab::VTabKind::Default;

    fn create(
        db: &mut VTabConnection,
        aux: Option<&Self::Aux>,
        args: &[&[u8]],
    ) -> rusqlite::Result<(String, Self)> {
        // For vec0, create and connect are the same
        Self::connect(db, aux, args)
    }

    fn destroy(&self) -> rusqlite::Result<()> {
        // No cleanup needed for in-memory table
        Ok(())
    }
}

impl<'vtab> UpdateVTab<'vtab> for Vec0Tab {
    fn delete(&mut self, arg: rusqlite::types::ValueRef<'_>) -> rusqlite::Result<()> {
        let rowid = arg.as_i64()?;
        self.rows.retain(|row| row.rowid != rowid);
        Ok(())
    }

    fn insert(&mut self, args: &Values<'_>) -> rusqlite::Result<i64> {
        // args[0]: NULL for auto-rowid
        // args[1]: new rowid (or NULL for auto)
        // args[2..]: column values

        let new_rowid = if args.len() > 1 {
            // Try to get rowid, or auto-generate
            args.get::<Option<i64>>(1)?
                .unwrap_or_else(|| self.rows.iter().map(|r| r.rowid).max().unwrap_or(0) + 1)
        } else {
            1
        };

        let mut data = Vec::new();
        for i in 2..args.len() {
            // Try to get as blob first (vectors), then as string
            if let Ok(blob) = args.get::<Vec<u8>>(i) {
                data.push(Some(blob));
            } else if let Ok(text) = args.get::<String>(i) {
                data.push(Some(text.into_bytes()));
            } else {
                data.push(None);
            }
        }

        self.rows.push(Row {
            rowid: new_rowid,
            data,
        });

        Ok(new_rowid)
    }

    fn update(&mut self, args: &Values<'_>) -> rusqlite::Result<()> {
        // args[0]: old rowid
        // args[1]: new rowid
        // args[2..]: new column values

        let old_rowid = args.get::<i64>(0)?;
        let new_rowid = args.get::<i64>(1)?;

        if let Some(row) = self.rows.iter_mut().find(|r| r.rowid == old_rowid) {
            row.rowid = new_rowid;
            row.data.clear();

            for i in 2..args.len() {
                // Try to get as blob first (vectors), then as string
                if let Ok(blob) = args.get::<Vec<u8>>(i) {
                    row.data.push(Some(blob));
                } else if let Ok(text) = args.get::<String>(i) {
                    row.data.push(Some(text.into_bytes()));
                } else {
                    row.data.push(None);
                }
            }
        }

        Ok(())
    }
}

/// vec0 cursor for iteration
#[repr(C)]
pub struct Vec0TabCursor<'vtab> {
    base: sqlite3_vtab_cursor,
    phantom: PhantomData<&'vtab Vec0Tab>,
    current_row: usize,
    row_count: usize,
}

impl<'vtab> Vec0TabCursor<'vtab> {
    fn new(table: &Vec0Tab) -> Self {
        Vec0TabCursor {
            base: sqlite3_vtab_cursor::default(),
            phantom: PhantomData,
            current_row: 0,
            row_count: table.rows.len(),
        }
    }

    /// Accessor to the associated virtual table
    fn vtab(&self) -> &Vec0Tab {
        unsafe { &*(self.base.pVtab as *const Vec0Tab) }
    }
}

unsafe impl VTabCursor for Vec0TabCursor<'_> {
    fn filter(
        &mut self,
        _idx_num: c_int,
        _idx_str: Option<&str>,
        _args: &Values<'_>,
    ) -> rusqlite::Result<()> {
        // Reset to beginning
        self.current_row = 0;
        Ok(())
    }

    fn next(&mut self) -> rusqlite::Result<()> {
        self.current_row += 1;
        Ok(())
    }

    fn eof(&self) -> bool {
        self.current_row >= self.row_count
    }

    fn column(&self, ctx: &mut Context, col: c_int) -> rusqlite::Result<()> {
        let table = self.vtab();

        if self.current_row >= table.rows.len() {
            ctx.set_result(&rusqlite::types::Null)?;
            return Ok(());
        }

        let row = &table.rows[self.current_row];
        let col_idx = col as usize;

        if col_idx < row.data.len() {
            match &row.data[col_idx] {
                Some(data) => ctx.set_result(&data.as_slice())?,
                None => ctx.set_result(&rusqlite::types::Null)?,
            }
        } else {
            ctx.set_result(&rusqlite::types::Null)?;
        }

        Ok(())
    }

    fn rowid(&self) -> rusqlite::Result<i64> {
        let table = self.vtab();

        if self.current_row < table.rows.len() {
            Ok(table.rows[self.current_row].rowid)
        } else {
            Ok(0)
        }
    }
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
    fn test_register_vec0_works() {
        let db = Connection::open_in_memory().unwrap();
        let result = register_vec0_module(&db);
        assert!(result.is_ok());
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
