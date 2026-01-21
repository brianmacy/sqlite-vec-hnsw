//! Build script for sqlite-vec-hnsw
//!
//! When compiled with `loadable_extension_alias` feature, this generates an additional
//! entry point alias if SQLITE_VEC_ENTRY_POINT_ALIAS environment variable is set.

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Only generate alias code if the feature is enabled
    if env::var("CARGO_FEATURE_LOADABLE_EXTENSION_ALIAS").is_ok() {
        let out_dir = env::var("OUT_DIR").unwrap();
        let dest_path = Path::new(&out_dir).join("entry_point_alias.rs");

        let code = if let Ok(alias_name) = env::var("SQLITE_VEC_ENTRY_POINT_ALIAS") {
            // Validate the alias name looks like a valid C identifier
            if alias_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                && !alias_name.is_empty()
                && !alias_name.chars().next().unwrap().is_numeric()
            {
                format!(
                    r#"
/// Additional entry point alias generated at build time
/// via SQLITE_VEC_ENTRY_POINT_ALIAS environment variable
#[unsafe(no_mangle)]
pub unsafe extern "C" fn {alias_name}(
    db: *mut rusqlite::ffi::sqlite3,
    err_msg: *mut *mut std::os::raw::c_char,
    api: *mut rusqlite::ffi::sqlite3_api_routines,
) -> std::os::raw::c_int {{
    // SAFETY: This is just a forwarding call to the main entry point
    unsafe {{ crate::sqlite3_sqlitevechnsw_init(db, err_msg, api) }}
}}
"#
                )
            } else {
                eprintln!(
                    "cargo:warning=Invalid SQLITE_VEC_ENTRY_POINT_ALIAS: {}",
                    alias_name
                );
                String::new()
            }
        } else {
            String::new()
        };

        fs::write(&dest_path, code).unwrap();

        // Tell cargo to rerun if this env var changes
        println!("cargo:rerun-if-env-changed=SQLITE_VEC_ENTRY_POINT_ALIAS");
    }

    // Always rerun if the feature flag changes
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LOADABLE_EXTENSION_ALIAS");
}
