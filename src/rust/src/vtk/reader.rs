// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-threaded VTK file reader using Rayon.

use rayon::prelude::*;
use std::fs;
use std::path::Path;
use thiserror::Error;

use super::mesh::{ArrayFilter, MeshArrays};
use super::parser::{parse_vtk_xml, VTKParseError};

/// Errors that can occur during VTK file reading.
#[derive(Error, Debug)]
pub enum VTKReadError {
    /// I/O error while reading the file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Error parsing the VTK content.
    #[error("Parse error: {0}")]
    Parse(#[from] VTKParseError),
}

/// Read a single VTK file and parse it.
///
/// # Arguments
///
/// * `path` - Path to the VTK file
/// * `filter` - Array include/exclude filter
/// * `skip_cells` - If true, skip cell topology and cell data
/// * `skip_point_data` - If true, skip point data field arrays
///
/// # Returns
///
/// A `MeshArrays` structure populated with the parsed data.
pub fn read_vtk_file_raw<P: AsRef<Path>>(
    path: P,
    filter: &ArrayFilter,
    skip_cells: bool,
    skip_point_data: bool,
) -> Result<MeshArrays, VTKReadError> {
    let raw = fs::read(path.as_ref())?;
    let mesh = parse_vtk_xml(&raw, filter, skip_cells, skip_point_data)?;
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
/// * `filter` - Array include/exclude filter (shared across all files)
/// * `skip_cells` - If true, skip cell topology and cell data
/// * `skip_point_data` - If true, skip point data field arrays
pub fn read_vtk_files_parallel_raw<P: AsRef<Path> + Sync>(
    paths: &[P],
    filter: &ArrayFilter,
    skip_cells: bool,
    skip_point_data: bool,
) -> Vec<Result<MeshArrays, VTKReadError>> {
    paths
        .par_iter()
        .map(|p| read_vtk_file_raw(p, filter, skip_cells, skip_point_data))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
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

    fn no_filter() -> ArrayFilter {
        ArrayFilter::new(None, None)
    }

    #[test]
    fn test_read_single_file() {
        let dir = tempdir().unwrap();
        let path = create_test_vtu(dir.path(), "test.vtu");
        let mesh = read_vtk_file_raw(&path, &no_filter(), false, false).unwrap();
        assert_eq!(mesh.n_points, 3);
        assert_eq!(mesh.n_cells, 1);
    }

    #[test]
    fn test_read_parallel() {
        let dir = tempdir().unwrap();
        let paths: Vec<_> = (0..4)
            .map(|i| create_test_vtu(dir.path(), &format!("test_{i}.vtu")))
            .collect();

        let results = read_vtk_files_parallel_raw(&paths, &no_filter(), false, false);
        assert_eq!(results.len(), 4);
        for result in results {
            let mesh = result.unwrap();
            assert_eq!(mesh.n_points, 3);
        }
    }

    #[test]
    fn test_file_not_found() {
        let result = read_vtk_file_raw("/nonexistent/path/file.vtu", &no_filter(), false, false);
        assert!(matches!(result, Err(VTKReadError::Io(_))));
    }
}
