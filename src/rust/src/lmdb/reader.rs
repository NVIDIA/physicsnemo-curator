// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Low-level LMDB reader for ASE database files.
//!
//! Handles opening the LMDB environment, iterating over data rows
//! (skipping reserved keys), and decompressing/parsing each value
//! from zlib-compressed JSON into [`serde_json::Value`].

use flate2::read::ZlibDecoder;
use rayon::prelude::*;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

/// Reserved LMDB keys that are not data rows.
const RESERVED_KEYS: &[&str] = &["nextid", "deleted_ids", "metadata"];

/// Errors that can occur during LMDB reading.
#[derive(Error, Debug)]
pub enum LmdbReadError {
    /// Error opening or reading the LMDB environment.
    #[error("LMDB error: {0}")]
    Lmdb(#[from] heed::Error),
    /// Zlib decompression failed.
    #[error("Decompression error: {0}")]
    Decompress(#[from] std::io::Error),
    /// JSON parsing failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// UTF-8 decoding failed.
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

/// A single row from an ASE LMDB database.
///
/// Contains the integer row ID and the parsed JSON value (a dict
/// with `__ndarray__` markers still present as JSON objects).
#[derive(Debug)]
pub struct LmdbRow {
    /// The integer row ID (1-based, as stored in ASE LMDB).
    pub id: i64,
    /// The parsed JSON value for this row.
    pub data: serde_json::Value,
}

/// Decompress a zlib-compressed byte slice and parse as JSON.
fn decompress_and_parse(compressed: &[u8]) -> Result<serde_json::Value, LmdbReadError> {
    let mut decoder = ZlibDecoder::new(compressed);
    let mut json_bytes = Vec::new();
    decoder.read_to_end(&mut json_bytes)?;
    let value: serde_json::Value = serde_json::from_slice(&json_bytes)?;
    Ok(value)
}

/// Read all data rows from a single `.aselmdb` file.
///
/// Opens the LMDB environment in read-only mode, iterates over all
/// key-value pairs, skips reserved keys (`nextid`, `deleted_ids`,
/// `metadata`), and returns the parsed rows sorted by ID.
///
/// # Arguments
///
/// * `path` - Path to the `.aselmdb` file
///
/// # Returns
///
/// A vector of [`LmdbRow`] sorted by ascending ID.
///
/// # Errors
///
/// Returns [`LmdbReadError`] if the file cannot be opened, a value
/// cannot be decompressed, or JSON parsing fails.
pub fn read_lmdb_rows<P: AsRef<Path>>(path: P) -> Result<Vec<LmdbRow>, LmdbReadError> {
    let path = path.as_ref();

    // Open LMDB in read-only mode with no-sub-dir (single file, not directory).
    let env = unsafe {
        heed::EnvOpenOptions::new()
            .flags(heed::EnvFlags::NO_SUB_DIR | heed::EnvFlags::READ_ONLY)
            .open(path)?
    };

    let rtxn = env.read_txn()?;
    let db: heed::Database<heed::types::Bytes, heed::types::Bytes> = env
        .open_database(&rtxn, None)?
        .expect("default database must exist");

    let mut rows = Vec::new();

    for result in db.iter(&rtxn)? {
        let (key_bytes, val_bytes) = result?;

        // Keys are ASCII-encoded strings.
        let key_str = std::str::from_utf8(key_bytes)?;

        // Skip reserved (non-data) keys.
        if RESERVED_KEYS.contains(&key_str) {
            continue;
        }

        // Data keys are integer IDs encoded as ASCII strings.
        let id: i64 = match key_str.parse() {
            Ok(id) => id,
            Err(_) => continue, // Skip any unknown non-integer keys.
        };

        let data = decompress_and_parse(val_bytes)?;
        rows.push(LmdbRow { id, data });
    }

    rtxn.commit()?;

    // Sort by ID for deterministic ordering.
    rows.sort_by_key(|r| r.id);

    Ok(rows)
}

/// Read rows from multiple `.aselmdb` files in parallel using Rayon.
///
/// Each file is read independently on a Rayon thread. Results are
/// returned in the same order as the input paths.
///
/// # Arguments
///
/// * `paths` - Slice of paths to `.aselmdb` files
///
/// # Returns
///
/// A vector of results, one per input path, in the same order.
pub fn read_lmdb_files_parallel<P: AsRef<Path> + Sync>(
    paths: &[P],
) -> Vec<Result<Vec<LmdbRow>, LmdbReadError>> {
    paths.par_iter().map(read_lmdb_rows).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    /// Create a minimal `.aselmdb` file with the given rows for testing.
    ///
    /// Writes `nextid`, `deleted_ids`, and the data rows as
    /// zlib-compressed JSON, mimicking the ase-db-backends format.
    fn create_test_aselmdb(dir: &Path, name: &str, rows: &[(i64, &str)]) -> std::path::PathBuf {
        let db_path = dir.join(name);

        let env = unsafe {
            heed::EnvOpenOptions::new()
                .flags(heed::EnvFlags::NO_SUB_DIR)
                .map_size(10 * 1024 * 1024) // 10 MB
                .open(&db_path)
                .expect("failed to open LMDB env")
        };

        let mut wtxn = env.write_txn().expect("failed to begin write txn");
        let db: heed::Database<heed::types::Bytes, heed::types::Bytes> = env
            .create_database(&mut wtxn, None)
            .expect("failed to create db");

        // Helper to zlib-compress a string.
        let compress = |s: &str| -> Vec<u8> {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(s.as_bytes()).unwrap();
            encoder.finish().unwrap()
        };

        // Write nextid.
        let next_id = rows.iter().map(|(id, _)| id).max().unwrap_or(&0) + 1;
        db.put(&mut wtxn, b"nextid", &compress(&next_id.to_string()))
            .unwrap();

        // Write deleted_ids (empty list).
        db.put(&mut wtxn, b"deleted_ids", &compress("[]")).unwrap();

        // Write data rows.
        for (id, json_str) in rows {
            let key = id.to_string();
            db.put(&mut wtxn, key.as_bytes(), &compress(json_str))
                .unwrap();
        }

        wtxn.commit().expect("failed to commit");

        // Fully close the env and release the heed cache entry so that
        // `read_lmdb_rows` can reopen the same path with READ_ONLY flags.
        env.prepare_for_closing().wait();

        db_path
    }

    #[test]
    fn test_read_empty_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = create_test_aselmdb(dir.path(), "empty.aselmdb", &[]);
        let rows = read_lmdb_rows(&path).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_read_single_row() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{"numbers": [1, 6, 8], "positions": {"__ndarray__": [[3, 3], "float64", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]}}"#;
        let path = create_test_aselmdb(dir.path(), "single.aselmdb", &[(1, json)]);

        let rows = read_lmdb_rows(&path).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, 1);

        // Verify the parsed JSON structure.
        let data = &rows[0].data;
        assert!(data.is_object());
        assert!(data["numbers"].is_array());
        assert!(data["positions"].is_object());
        assert!(data["positions"]["__ndarray__"].is_array());
    }

    #[test]
    fn test_read_multiple_rows_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let rows_data = [
            (3, r#"{"numbers": [8]}"#),
            (1, r#"{"numbers": [1]}"#),
            (2, r#"{"numbers": [6]}"#),
        ];
        let path = create_test_aselmdb(dir.path(), "multi.aselmdb", &rows_data);

        let rows = read_lmdb_rows(&path).unwrap();
        assert_eq!(rows.len(), 3);
        // Verify sorted by ID.
        assert_eq!(rows[0].id, 1);
        assert_eq!(rows[1].id, 2);
        assert_eq!(rows[2].id, 3);
    }

    #[test]
    fn test_skips_reserved_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = create_test_aselmdb(
            dir.path(),
            "reserved.aselmdb",
            &[(1, r#"{"numbers": [1]}"#)],
        );

        let rows = read_lmdb_rows(&path).unwrap();
        // Should only contain the data row, not nextid or deleted_ids.
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, 1);
    }

    #[test]
    fn test_parallel_read() {
        let dir = tempfile::tempdir().unwrap();
        let paths: Vec<_> = (0..4)
            .map(|i| {
                let json = format!(r#"{{"numbers": [{}]}}"#, i);
                create_test_aselmdb(dir.path(), &format!("db_{}.aselmdb", i), &[(1, &json)])
            })
            .collect();

        let results = read_lmdb_files_parallel(&paths);
        assert_eq!(results.len(), 4);
        for result in results {
            let rows = result.unwrap();
            assert_eq!(rows.len(), 1);
        }
    }

    #[test]
    fn test_ndarray_marker_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{"cell": {"__ndarray__": [[3, 3], "float64", [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]]}}"#;
        let path = create_test_aselmdb(dir.path(), "ndarray.aselmdb", &[(1, json)]);

        let rows = read_lmdb_rows(&path).unwrap();
        let cell = &rows[0].data["cell"]["__ndarray__"];
        assert!(cell.is_array());

        // Shape should be [3, 3].
        let shape = &cell[0];
        assert_eq!(shape[0], 3);
        assert_eq!(shape[1], 3);

        // Dtype should be "float64".
        assert_eq!(cell[1], "float64");

        // Data should have 9 elements.
        let data = cell[2].as_array().unwrap();
        assert_eq!(data.len(), 9);
    }

    #[test]
    fn test_file_not_found() {
        let result = read_lmdb_rows("/nonexistent/path/file.aselmdb");
        assert!(result.is_err());
    }
}
