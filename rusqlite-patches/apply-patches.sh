#!/bin/bash
set -e

echo "🔧 Rusqlite Custom Features Setup"
echo "=================================="

# Configuration
RUSQLITE_DIR="$HOME/open_dev/rusqlite"
PATCH_DIR="$(dirname "$0")"
BRANCH="custom-vtab-features"

# Step 1: Clone if needed
if [ ! -d "$RUSQLITE_DIR" ]; then
    echo "📦 Cloning rusqlite fork..."
    cd "$HOME/open_dev"
    git clone git@github.com:brianmacy/rusqlite.git
else
    echo "✓ Rusqlite already cloned at $RUSQLITE_DIR"
fi

cd "$RUSQLITE_DIR"

# Step 2: Create branch
echo "🌿 Creating branch: $BRANCH"
git checkout main || git checkout master
git pull origin main || git pull origin master
git checkout -b "$BRANCH" || git checkout "$BRANCH"

# Step 3: Apply changes manually (patches are conceptual guides)
echo "✏️  Applying changes to rusqlite..."

# Change 1: Add overload_function to src/lib.rs
echo "  → Adding Connection::overload_function()..."
cat > /tmp/overload_function.rs << 'EOF'

    /// Register a custom operator for virtual tables
    ///
    /// This is typically used to enable MATCH operator for custom virtual tables.
    /// For example, FTS5 uses this to register MATCH.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use rusqlite::{Connection, Result};
    /// # fn main() -> Result<()> {
    /// let db = Connection::open_in_memory()?;
    /// db.overload_function("match", 2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn overload_function(&self, name: &str, n_arg: c_int) -> Result<()> {
        let c_name = std::ffi::CString::new(name)?;
        unsafe {
            self.db.borrow_mut().check(ffi::sqlite3_overload_function(
                self.db.borrow().db(),
                c_name.as_ptr(),
                n_arg,
            ))
        }
    }
EOF

# Find the right place to insert (after create_scalar_function, before last_insert_rowid)
if ! grep -q "fn overload_function" src/lib.rs; then
    # Insert after the create_collation functions, before get_aux
    sed -i.bak '/pub fn get_aux<T: Send/i\
\
    /// Register a custom operator for virtual tables\
    ///\
    /// This is typically used to enable MATCH operator for custom virtual tables.\
    /// For example, FTS5 uses this to register MATCH.\
    ///\
    /// # Example\
    /// ```rust,no_run\
    /// # use rusqlite::{Connection, Result};\
    /// # fn main() -> Result<()> {\
    /// let db = Connection::open_in_memory()?;\
    /// db.overload_function("match", 2)?;\
    /// # Ok(())\
    /// # }\
    /// ```\
    pub fn overload_function(\&self, name: \&str, n_arg: c_int) -> Result<()> {\
        let c_name = std::ffi::CString::new(name)?;\
        unsafe {\
            self.db.borrow_mut().check(ffi::sqlite3_overload_function(\
                self.db.borrow().db(),\
                c_name.as_ptr(),\
                n_arg,\
            ))\
        }\
    }\
' src/lib.rs
    echo "    ✓ Added overload_function()"
else
    echo "    ✓ overload_function() already exists"
fi

# Change 2: Add get_connection to src/functions.rs
echo "  → Adding Context::get_connection()..."
if ! grep -q "fn get_connection" src/functions.rs; then
    # Add after the existing Context methods
    sed -i.bak '/pub fn set_aux<T: Send/a\
\
    /// Get the database connection handle\
    ///\
    /// This allows scalar functions to execute queries or modify the database.\
    /// Use with caution as this breaks the typical stateless nature of SQL functions.\
    ///\
    /// # Safety\
    /// The returned connection borrows from this Context and must not outlive it.\
    /// The caller must ensure no conflicting database operations occur.\
    ///\
    /// # Example\
    /// ```rust,no_run\
    /// # use rusqlite::{Connection, Result, functions::FunctionFlags};\
    /// # fn main() -> Result<()> {\
    /// # let db = Connection::open_in_memory()?;\
    /// db.create_scalar_function("my_func", 1, FunctionFlags::default(), |ctx| {\
    ///     let conn = unsafe { ctx.get_connection()? };\
    ///     // Can now execute queries\
    ///     Ok(())\
    /// })?;\
    /// # Ok(())\
    /// # }\
    /// ```\
    pub unsafe fn get_connection(\&self) -> Result<Connection> {\
        Connection::from_handle(ffi::sqlite3_context_db_handle(self.ctx))\
    }\
' src/functions.rs
    echo "    ✓ Added get_connection()"
else
    echo "    ✓ get_connection() already exists"
fi

# Change 3: Add integrity method to VTab trait in src/vtab.rs
echo "  → Adding VTab::integrity()..."
if ! grep -q "fn integrity" src/vtab.rs; then
    # This is more complex - need to add both trait method and C callback
    echo "    ⚠️  Manual edit needed for VTab::integrity() - see rusqlite-patches/03-vtab-integrity.patch"
    echo "    Add to VTab trait:"
    echo "    fn integrity(&self, _schema: &str, _table_name: &str, _flags: c_int) -> Result<Option<String>> { Ok(None) }"
else
    echo "    ✓ integrity() already exists"
fi

# Step 4: Build and test
echo ""
echo "🔨 Building rusqlite with changes..."
cargo build --features "bundled vtab blob functions"

echo ""
echo "🧪 Running rusqlite tests..."
cargo test --features "bundled vtab blob functions"

echo ""
echo "✅ Changes applied successfully!"
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Commit: git add -A && git commit -m 'Add virtual table enhancements'"
echo "3. Push: git push origin $BRANCH"
echo "4. Test with sqlite-vec-hnsw: cd ~/open_dev/sqlite-vec-hnsw && cargo build"

