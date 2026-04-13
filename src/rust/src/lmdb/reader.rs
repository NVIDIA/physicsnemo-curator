// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Low-level LMDB reader for ASE database files.
//!
//! Handles opening the LMDB environment, iterating over data rows
//! (skipping reserved keys), and decompressing/parsing each value
//! from zlib-compressed JSON into [`serde_json::Value`].
//!
//! ## Performance
//!
//! - **Intra-file parallelism**: After a fast sequential cursor scan to
//!   collect raw compressed bytes, the expensive decompress + JSON parse
//!   step runs in parallel across rows via Rayon.
//! - **Inter-file parallelism**: Multiple `.aselmdb` files are read
//!   concurrently with `read_lmdb_files_parallel`.
//! - **Buffer estimation**: Decompression pre-allocates output buffers
//!   at 4× the compressed size to reduce reallocations.

use flate2::read::ZlibDecoder;
use rayon::prelude::*;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

/// Reserved LMDB keys that are not data rows.
const RESERVED_KEYS: &[&str] = &["nextid", "deleted_ids", "metadata"];

/// Heuristic multiplier for estimating decompressed size from compressed size.
/// JSON text typically compresses to ~25% of its original size with zlib.
const DECOMPRESS_SIZE_MULTIPLIER: usize = 4;

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

// Allow converting (id, LmdbReadError) from Rayon back to LmdbReadError.
// The id context is only used for parallel error propagation.
impl From<(i64, LmdbReadError)> for LmdbReadError {
    fn from(pair: (i64, LmdbReadError)) -> Self {
        pair.1
    }
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

/// Raw key-value pair extracted from the LMDB cursor before decompression.
struct RawEntry {
    /// The integer row ID.
    id: i64,
    /// The compressed bytes (zlib-compressed JSON).
    compressed: Vec<u8>,
}

/// Decompress a zlib-compressed byte slice and parse as JSON.
///
/// Pre-allocates the output buffer based on `compressed_len` to reduce
/// reallocations during decompression.
fn decompress_and_parse(compressed: &[u8]) -> Result<serde_json::Value, LmdbReadError> {
    let estimated_size = compressed.len() * DECOMPRESS_SIZE_MULTIPLIER;
    let mut decoder = ZlibDecoder::new(compressed);
    let mut json_bytes = Vec::with_capacity(estimated_size);
    decoder.read_to_end(&mut json_bytes)?;
    let value: serde_json::Value = serde_json::from_slice(&json_bytes)?;
    Ok(value)
}

/// Read all data rows from a single `.aselmdb` file.
///
/// Opens the LMDB environment in read-only mode, performs a fast
/// sequential cursor scan to collect raw compressed bytes, then
/// decompresses and parses rows **in parallel** via Rayon.
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

    // --- Phase 1: Fast sequential LMDB cursor scan ---
    // The LMDB cursor is inherently single-threaded (memory-mapped I/O).
    // We collect the raw compressed bytes as quickly as possible, then
    // release the read transaction before doing any heavy processing.
    let raw_entries = {
        let env = unsafe {
            heed::EnvOpenOptions::new()
                .flags(heed::EnvFlags::NO_SUB_DIR | heed::EnvFlags::READ_ONLY)
                .open(path)?
        };

        let rtxn = env.read_txn()?;
        let db: heed::Database<heed::types::Bytes, heed::types::Bytes> = env
            .open_database(&rtxn, None)?
            .expect("default database must exist");

        let mut entries = Vec::new();

        for result in db.iter(&rtxn)? {
            let (key_bytes, val_bytes) = result?;

            let key_str = std::str::from_utf8(key_bytes)?;

            // Skip reserved (non-data) keys.
            if RESERVED_KEYS.contains(&key_str) {
                continue;
            }

            // Data keys are integer IDs encoded as ASCII strings.
            let id: i64 = match key_str.parse() {
                Ok(id) => id,
                Err(_) => continue,
            };

            // Copy compressed bytes out of the memory-mapped region so we
            // can release the read transaction before parallel decompression.
            entries.push(RawEntry {
                id,
                compressed: val_bytes.to_vec(),
            });
        }

        rtxn.commit()?;
        entries
    };

    // --- Phase 2: Parallel decompress + JSON parse ---
    // Each row is decompressed and parsed independently on a Rayon thread.
    let mut rows: Vec<LmdbRow> = raw_entries
        .into_par_iter()
        .map(|entry| {
            let data = decompress_and_parse(&entry.compressed).map_err(|e| (entry.id, e))?;
            Ok(LmdbRow { id: entry.id, data })
        })
        .collect::<Result<Vec<_>, (i64, LmdbReadError)>>()?;

    // Sort by ID for deterministic ordering.
    rows.sort_unstable_by_key(|r| r.id);

    Ok(rows)
}

/// Read rows from multiple `.aselmdb` files in parallel using Rayon.
///
/// Each file is read independently on a Rayon thread.  Within each
/// file, rows are also decompressed in parallel (nested Rayon
/// parallelism).  Results are returned in the same order as the
/// input paths.
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
    /// zlib-compressed JSON, mimicking the ASE LMDB format.
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

    #[test]
    fn test_intra_file_parallelism_many_rows() {
        // Verify correctness with enough rows that Rayon actually
        // spawns parallel work items (typically > 4).
        let dir = tempfile::tempdir().unwrap();
        let rows_data: Vec<(i64, String)> = (1..=32)
            .map(|i| {
                (
                    i,
                    format!(
                        r#"{{"numbers": [{}], "positions": {{"__ndarray__": [[1, 3], "float64", [{}.0, 0.0, 0.0]]}}}}"#,
                        i, i
                    ),
                )
            })
            .collect();
        let rows_refs: Vec<(i64, &str)> =
            rows_data.iter().map(|(id, s)| (*id, s.as_str())).collect();
        let path = create_test_aselmdb(dir.path(), "many.aselmdb", &rows_refs);

        let rows = read_lmdb_rows(&path).unwrap();
        assert_eq!(rows.len(), 32);
        // Verify sorted and correct IDs.
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.id, (i as i64) + 1);
        }
    }
}
