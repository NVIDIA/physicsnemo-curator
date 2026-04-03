// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! VTK XML format parser using quick-xml.

use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::BufRead;
use thiserror::Error;

use super::mesh::{DataArray, VTKMesh};

/// Errors that can occur during VTK parsing.
#[derive(Error, Debug)]
pub enum VTKParseError {
    /// XML parsing error.
    #[error("XML parsing error: {0}")]
    Xml(#[from] quick_xml::Error),
    /// Invalid VTK format.
    #[error("Invalid VTK format: {0}")]
    InvalidFormat(String),
    /// UTF-8 encoding error.
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

/// Data format in VTK XML.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DataFormat {
    Ascii,
    Binary,
    Appended,
}

/// Data type in VTK XML.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
}

impl DataType {
    #[allow(dead_code)]
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "Float32" => Some(DataType::Float32),
            "Float64" => Some(DataType::Float64),
            "Int32" => Some(DataType::Int32),
            "Int64" => Some(DataType::Int64),
            "UInt8" => Some(DataType::UInt8),
            _ => None,
        }
    }
}

/// Metadata for a DataArray element.
struct DataArrayInfo {
    name: String,
    num_components: usize,
    format: DataFormat,
}

fn parse_data_array_attrs(
    e: &quick_xml::events::BytesStart,
) -> Result<DataArrayInfo, VTKParseError> {
    let mut name = String::new();
    let mut num_components = 1usize;
    let mut format = DataFormat::Ascii;

    for attr in e.attributes().flatten() {
        let key = std::str::from_utf8(attr.key.as_ref())?;
        let val = std::str::from_utf8(&attr.value)?;
        match key {
            "Name" => name = val.to_string(),
            "NumberOfComponents" => num_components = val.parse().unwrap_or(1),
            "format" => {
                format = match val {
                    "ascii" => DataFormat::Ascii,
                    "binary" => DataFormat::Binary,
                    "appended" => DataFormat::Appended,
                    _ => DataFormat::Ascii,
                }
            }
            _ => {}
        }
    }

    Ok(DataArrayInfo {
        name,
        num_components,
        format,
    })
}

fn parse_ascii_floats(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect()
}

fn parse_ascii_ints(text: &str) -> Vec<i64> {
    text.split_whitespace()
        .filter_map(|s| s.parse::<i64>().ok())
        .collect()
}

fn parse_ascii_u8(text: &str) -> Vec<u8> {
    text.split_whitespace()
        .filter_map(|s| s.parse::<u8>().ok())
        .collect()
}

/// Parse a VTK XML file from a reader.
///
/// # Arguments
///
/// * `reader` - A buffered reader containing VTK XML data
/// * `format` - The VTK format type ("vtu" or "vtp")
///
/// # Returns
///
/// A `VTKMesh` structure populated with the parsed data.
///
/// # Errors
///
/// Returns `VTKParseError` if the XML is malformed or contains invalid data.
pub fn parse_vtk<R: BufRead>(reader: R, format: &str) -> Result<VTKMesh, VTKParseError> {
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.config_mut().trim_text(true);

    let mut mesh = VTKMesh::new(format);
    let mut buf = Vec::new();
    let mut current_section: Option<String> = None;
    let mut in_point_data = false;
    let mut in_cell_data = false;

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())?;
                match name {
                    "Piece" => {
                        for attr in e.attributes().flatten() {
                            let key = std::str::from_utf8(attr.key.as_ref())?;
                            let val = std::str::from_utf8(&attr.value)?;
                            match key {
                                "NumberOfPoints" => mesh.n_points = val.parse().unwrap_or(0),
                                "NumberOfCells" | "NumberOfPolys" | "NumberOfVerts" => {
                                    mesh.n_cells += val.parse().unwrap_or(0);
                                }
                                _ => {}
                            }
                        }
                    }
                    "Points" => current_section = Some("Points".to_string()),
                    "Cells" | "Polys" | "Verts" => current_section = Some("Cells".to_string()),
                    "PointData" => in_point_data = true,
                    "CellData" => in_cell_data = true,
                    "DataArray" => {
                        let array_info = parse_data_array_attrs(&e)?;
                        if array_info.format == DataFormat::Ascii {
                            let text = xml_reader.read_text(e.name())?;
                            store_data_array(
                                &mut mesh,
                                &current_section,
                                in_point_data,
                                in_cell_data,
                                &array_info,
                                &text,
                            );
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())?;
                match name {
                    "Points" | "Cells" | "Polys" | "Verts" => current_section = None,
                    "PointData" => in_point_data = false,
                    "CellData" => in_cell_data = false,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(VTKParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    // Generate cell types if not present (VTP files use implicit types)
    if mesh.types.is_empty() && mesh.n_cells > 0 {
        mesh.types = vec![super::cell_types::VTK_TRIANGLE; mesh.n_cells];
    }

    Ok(mesh)
}

fn store_data_array(
    mesh: &mut VTKMesh,
    section: &Option<String>,
    in_point_data: bool,
    in_cell_data: bool,
    info: &DataArrayInfo,
    text: &str,
) {
    if let Some(sec) = section {
        match sec.as_str() {
            "Points" => {
                mesh.points = parse_ascii_floats(text);
            }
            "Cells" => match info.name.as_str() {
                "connectivity" | "" => {
                    mesh.connectivity = parse_ascii_ints(text);
                }
                "offsets" => {
                    mesh.offsets = parse_ascii_ints(text);
                }
                "types" => {
                    mesh.types = parse_ascii_u8(text);
                }
                _ => {}
            },
            _ => {}
        }
    } else if in_point_data && !info.name.is_empty() {
        let data = parse_ascii_floats(text);
        mesh.point_data.insert(
            info.name.clone(),
            DataArray {
                name: info.name.clone(),
                num_components: info.num_components,
                data,
            },
        );
    } else if in_cell_data && !info.name.is_empty() {
        let data = parse_ascii_floats(text);
        mesh.cell_data.insert(
            info.name.clone(),
            DataArray {
                name: info.name.clone(),
                num_components: info.num_components,
                data,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_simple_vtu() {
        let xml = r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  1 1 0  0 1 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          0 1 2 3
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          4
        </DataArray>
        <DataArray Name="types" type="UInt8" format="ascii">
          9
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#;

        let mesh = parse_vtk(Cursor::new(xml), "vtu").unwrap();
        assert_eq!(mesh.n_points, 4);
        assert_eq!(mesh.n_cells, 1);
        assert_eq!(mesh.points.len(), 12);
        assert_eq!(mesh.connectivity, vec![0, 1, 2, 3]);
        assert_eq!(mesh.offsets, vec![4]);
        assert_eq!(mesh.types, vec![9]);
    }

    #[test]
    fn test_parse_vtu_with_point_data() {
        let xml = r#"<?xml version="1.0"?>
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
      <PointData>
        <DataArray Name="Temperature" type="Float64" NumberOfComponents="1" format="ascii">
          100.0 200.0 300.0
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#;

        let mesh = parse_vtk(Cursor::new(xml), "vtu").unwrap();
        assert_eq!(mesh.n_points, 3);
        assert!(mesh.point_data.contains_key("Temperature"));
        let temp = &mesh.point_data["Temperature"];
        assert_eq!(temp.num_components, 1);
        assert_eq!(temp.data, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn test_parse_vtp_polydata() {
        let xml = r#"<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1">
  <PolyData>
    <Piece NumberOfPoints="3" NumberOfPolys="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0.5 1 0
        </DataArray>
      </Points>
      <Polys>
        <DataArray Name="connectivity" type="Int64" format="ascii">
          0 1 2
        </DataArray>
        <DataArray Name="offsets" type="Int64" format="ascii">
          3
        </DataArray>
      </Polys>
    </Piece>
  </PolyData>
</VTKFile>"#;

        let mesh = parse_vtk(Cursor::new(xml), "vtp").unwrap();
        assert_eq!(mesh.n_points, 3);
        assert_eq!(mesh.n_cells, 1);
        assert_eq!(mesh.format, "vtp");
        // VTP files get default triangle types
        assert_eq!(mesh.types, vec![5]);
    }
}
