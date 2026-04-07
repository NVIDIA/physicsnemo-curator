// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-threaded VTK file reader using Rayon.

use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use thiserror::Error;

use super::mesh::VTKMesh;
use super::parser::{parse_vtk, VTKParseError};

/// Errors that can occur during VTK file reading.
#[derive(Error, Debug)]
pub enum VTKReadError {
    /// I/O error while reading the file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Error parsing the VTK content.
    #[error("Parse error: {0}")]
    Parse(#[from] VTKParseError),
    /// File has an unknown or unsupported extension.
    #[error("Unknown file format: {0}")]
    UnknownFormat(String),
}

/// Determine VTK format from file extension.
fn get_vtk_format(path: &Path) -> Result<&'static str, VTKReadError> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("vtu") => Ok("vtu"),
        Some("vtp") => Ok("vtp"),
        Some("vtk") => Ok("vtk"),
        Some("vts") => Ok("vts"),
        Some("vtm") => Ok("vtm"),
        Some(ext) => Err(VTKReadError::UnknownFormat(ext.to_string())),
        None => Err(VTKReadError::UnknownFormat("no extension".to_string())),
    }
}

/// Read a single VTK file.
///
/// # Arguments
///
/// * `path` - Path to the VTK file
///
/// # Returns
///
/// A `VTKMesh` structure populated with the parsed data.
///
/// # Errors
///
/// Returns `VTKReadError` if the file cannot be read or parsed.
pub fn read_vtk_file<P: AsRef<Path>>(path: P) -> Result<VTKMesh, VTKReadError> {
    let path = path.as_ref();
    let format = get_vtk_format(path)?;
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mesh = parse_vtk(reader, format)?;
    Ok(mesh)
}

/// Read multiple VTK files in parallel using Rayon.
///
/// Returns a vector of results, one per file. Failed reads are
/// represented as errors in the result.
///
/// # Arguments
///
/// * `paths` - Slice of paths to VTK files
///
/// # Returns
///
/// A vector of results, one per input path, in the same order as the input.
pub fn read_vtk_files_parallel<P: AsRef<Path> + Sync>(
    paths: &[P],
) -> Vec<Result<VTKMesh, VTKReadError>> {
    paths.par_iter().map(read_vtk_file).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_vtu(dir: &Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        write!(
            file,
            r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0.5 1 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          0 1 2
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          3
        </DataArray>
        <DataArray Name="types" type="UInt8" format="ascii">
          5
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#
        )
        .unwrap();
        path
    }

    #[test]
    fn test_read_single_file() {
        let dir = tempdir().unwrap();
        let path = create_test_vtu(dir.path(), "test.vtu");
        let mesh = read_vtk_file(&path).unwrap();
        assert_eq!(mesh.n_points, 3);
        assert_eq!(mesh.n_cells, 1);
    }

    #[test]
    fn test_read_parallel() {
        let dir = tempdir().unwrap();
        let paths: Vec<_> = (0..4)
            .map(|i| create_test_vtu(dir.path(), &format!("test_{}.vtu", i)))
            .collect();

        let results = read_vtk_files_parallel(&paths);
        assert_eq!(results.len(), 4);
        for result in results {
            let mesh = result.unwrap();
            assert_eq!(mesh.n_points, 3);
        }
    }

    #[test]
    fn test_unknown_format() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.xyz");
        File::create(&path).unwrap();
        let result = read_vtk_file(&path);
        assert!(matches!(result, Err(VTKReadError::UnknownFormat(_))));
    }

    #[test]
    fn test_file_not_found() {
        let result = read_vtk_file("/nonexistent/path/file.vtu");
        assert!(matches!(result, Err(VTKReadError::Io(_))));
    }

    #[test]
    fn test_no_extension() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_file");
        File::create(&path).unwrap();
        let result = read_vtk_file(&path);
        assert!(matches!(result, Err(VTKReadError::UnknownFormat(ref s)) if s == "no extension"));
    }
}
