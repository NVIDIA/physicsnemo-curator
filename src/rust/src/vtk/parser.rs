// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! VTK XML parser with full binary, appended, and zlib support.
//!
//! Adapted from <https://github.com/coreyjadams/VtkToMesh> (Apache-2.0).
//! Handles ASCII, inline binary (base64 ± zlib), and appended data
//! (raw or base64, ± zlib compression).

use std::io::Read;

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use quick_xml::events::Event;
use quick_xml::name::QName;
use quick_xml::Reader;
use rayon::prelude::*;
use thiserror::Error;

use super::mesh::{ArrayFilter, DecodedArray, MeshArrays, ScalarType};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during VTK parsing.
#[derive(Error, Debug)]
pub enum VTKParseError {
    /// XML parsing error.
    #[error("XML parsing error: {0}")]
    Xml(#[from] quick_xml::Error),
    /// Invalid VTK format or data.
    #[error("Invalid VTK format: {0}")]
    InvalidFormat(String),
    /// UTF-8 encoding error.
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    /// I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Encoding info extracted from the VTKFile root element
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct EncodingInfo {
    header_type: ScalarType,
    compressor: Compressor,
}

#[derive(Clone, Copy, PartialEq)]
enum Compressor {
    None,
    ZLib,
}

impl EncodingInfo {
    fn header_bytes(&self) -> usize {
        self.header_type.byte_size()
    }
}

// ---------------------------------------------------------------------------
// Section tracking
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Section {
    None,
    PointData,
    CellData,
    Points,
    Cells,
    Verts,
    Lines,
    Strips,
    Polys,
    FieldData,
}

impl Section {
    fn is_cell_topo(self) -> bool {
        matches!(
            self,
            Section::Cells | Section::Verts | Section::Lines | Section::Strips | Section::Polys
        )
    }
}

// ---------------------------------------------------------------------------
// Array routing: where to store a decoded array
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum ArrayTarget {
    Points,
    Connectivity,
    Offsets,
    CellTypes,
    PointData,
    CellData,
}

fn array_target_for(name: &str, section: Section) -> Option<ArrayTarget> {
    match section {
        Section::Points => Some(ArrayTarget::Points),
        Section::PointData => Some(ArrayTarget::PointData),
        Section::CellData => Some(ArrayTarget::CellData),
        s if s.is_cell_topo() => match name {
            "connectivity" => Some(ArrayTarget::Connectivity),
            "offsets" => Some(ArrayTarget::Offsets),
            "types" => Some(ArrayTarget::CellTypes),
            _ => Option::None, // face_connectivity, face_offsets, etc.
        },
        _ => Option::None,
    }
}

// ---------------------------------------------------------------------------
// Deferred-decode structures
// ---------------------------------------------------------------------------

struct PendingArray {
    name: String,
    target: ArrayTarget,
    scalar_type: ScalarType,
    num_comp: usize,
    text: String,
}

struct AppendedEntry {
    name: String,
    target: ArrayTarget,
    scalar_type: ScalarType,
    num_comp: usize,
    offset: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum AppendedEncoding {
    Base64,
    Raw,
}

// ---------------------------------------------------------------------------
// DataArray attribute parsing
// ---------------------------------------------------------------------------

struct DaAttrs {
    name: String,
    scalar_type: ScalarType,
    num_comp: usize,
    is_appended: bool,
    appended_offset: usize,
    is_ascii: bool,
}

fn parse_da_attrs(
    e: &quick_xml::events::BytesStart<'_>,
) -> DaAttrs {
    let mut name = String::new();
    let mut type_str = String::new();
    let mut num_comp: usize = 1;
    let mut format_str = String::new();
    let mut offset: usize = 0;

    for attr in e.attributes().flatten() {
        let key = std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
        let val = std::str::from_utf8(&attr.value).unwrap_or("");
        match key {
            "Name" => name = val.to_string(),
            "type" => type_str = val.to_string(),
            "NumberOfComponents" => num_comp = val.parse().unwrap_or(1),
            "format" => format_str = val.to_string(),
            "offset" => offset = val.parse().unwrap_or(0),
            _ => {}
        }
    }

    DaAttrs {
        name,
        scalar_type: ScalarType::from_vtk_str(&type_str).unwrap_or(ScalarType::Float64),
        num_comp,
        is_appended: format_str == "appended",
        appended_offset: offset,
        is_ascii: format_str == "ascii",
    }
}

// ---------------------------------------------------------------------------
// Should-skip logic
// ---------------------------------------------------------------------------

fn should_skip(
    name: &str,
    section: Section,
    skip_cells: bool,
    skip_point_data: bool,
    filter: &ArrayFilter,
) -> bool {
    // Skip FieldData and unknown sections entirely
    if section == Section::None || section == Section::FieldData {
        return true;
    }
    // Skip ALL cell topology AND cell data when skip_cells is set
    if skip_cells && (section.is_cell_topo() || section == Section::CellData) {
        return true;
    }
    // Skip ALL point data fields when skip_point_data is set
    if skip_point_data && section == Section::PointData {
        return true;
    }
    // Points section (coordinates) is always kept
    if section == Section::Points {
        return false;
    }
    // Cell topology arrays are structural (kept unless skip_cells)
    if section.is_cell_topo() {
        return false;
    }
    // Apply user include/exclude filter to PointData / CellData field arrays
    if section == Section::PointData || section == Section::CellData {
        return !filter.should_include(name);
    }
    false
}

// ---------------------------------------------------------------------------
// ASCII text parsing helpers
// ---------------------------------------------------------------------------

fn parse_ascii_to_bytes(text: &str, scalar_type: ScalarType) -> Vec<u8> {
    match scalar_type {
        ScalarType::Float32 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::Float64 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::Int8 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<i8>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::Int16 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<i16>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::Int32 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::Int64 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<i64>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::UInt8 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<u8>().ok())
            .collect(),
        ScalarType::UInt16 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<u16>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::UInt32 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<u32>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        ScalarType::UInt64 => text
            .split_whitespace()
            .filter_map(|s| s.parse::<u64>().ok())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a VTK XML file from raw bytes.
///
/// Handles ASCII, inline binary (base64 ± zlib), and appended data
/// (raw or base64 ± zlib).
///
/// # Arguments
///
/// * `raw` - The entire file contents as a byte slice
/// * `filter` - Array include/exclude filter
/// * `skip_cells` - If `true`, skip all cell topology and cell data arrays
/// * `skip_point_data` - If `true`, skip all point data field arrays
///   (point *coordinates* are still read)
pub fn parse_vtk_xml(
    raw: &[u8],
    filter: &ArrayFilter,
    skip_cells: bool,
    skip_point_data: bool,
) -> Result<MeshArrays, VTKParseError> {
    let mut reader = Reader::from_reader(raw);
    reader.config_mut().trim_text(true);

    let mut encoding = EncodingInfo {
        header_type: ScalarType::UInt32,
        compressor: Compressor::None,
    };
    let mut section = Section::None;
    let mut n_points: usize = 0;
    let mut n_cells: usize = 0;

    // Deferred decode: collect raw text per kept array, decode after parsing
    let mut pending: Vec<PendingArray> = Vec::new();
    let mut cur: Option<(ArrayTarget, ScalarType, usize, String, bool)> = Option::None;
    // (target, type, ncomp, name, is_ascii)
    let mut text_buf = String::new();

    // Appended data tracking
    let mut appended_entries: Vec<AppendedEntry> = Vec::new();
    let mut appended_enc = AppendedEncoding::Base64;
    let mut appended_data_start: Option<usize> = Option::None;

    let mut buf = Vec::new();

    loop {
        let mut skip_to_end: Option<Vec<u8>> = Option::None;

        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let qname = e.name();
                let tag = std::str::from_utf8(qname.as_ref()).unwrap_or("");

                match tag {
                    "VTKFile" => {
                        for attr in e.attributes().flatten() {
                            let key =
                                std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
                            let val = std::str::from_utf8(&attr.value).unwrap_or("");
                            match key {
                                "header_type" => {
                                    if let Some(st) = ScalarType::from_vtk_str(val) {
                                        encoding.header_type = st;
                                    }
                                }
                                "compressor" if val.contains("ZLib") => {
                                    encoding.compressor = Compressor::ZLib;
                                }
                                _ => {}
                            }
                        }
                    }
                    "Piece" => {
                        for attr in e.attributes().flatten() {
                            let key =
                                std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
                            let val = std::str::from_utf8(&attr.value).unwrap_or("");
                            match key {
                                "NumberOfPoints" => {
                                    n_points = val.parse().unwrap_or(0);
                                }
                                "NumberOfCells" => {
                                    n_cells = val.parse().unwrap_or(0);
                                }
                                "NumberOfPolys" => {
                                    n_cells += val.parse::<usize>().unwrap_or(0);
                                }
                                "NumberOfVerts" | "NumberOfLines" | "NumberOfStrips" => {
                                    n_cells += val.parse::<usize>().unwrap_or(0);
                                }
                                _ => {}
                            }
                        }
                    }
                    "PointData" => section = Section::PointData,
                    "CellData" => section = Section::CellData,
                    "Points" => section = Section::Points,
                    "Cells" => section = Section::Cells,
                    "Verts" => section = Section::Verts,
                    "Lines" => section = Section::Lines,
                    "Strips" => section = Section::Strips,
                    "Polys" => section = Section::Polys,
                    "FieldData" => section = Section::FieldData,
                    "DataArray" => {
                        let da = parse_da_attrs(e);
                        let skip = should_skip(&da.name, section, skip_cells, skip_point_data, filter);

                        if da.is_appended {
                            if !skip {
                                if let Some(t) = array_target_for(&da.name, section) {
                                    appended_entries.push(AppendedEntry {
                                        name: da.name,
                                        target: t,
                                        scalar_type: da.scalar_type,
                                        num_comp: da.num_comp,
                                        offset: da.appended_offset,
                                    });
                                }
                            }
                        } else if skip {
                            skip_to_end = Some(qname.as_ref().to_vec());
                        } else if let Some(t) = array_target_for(&da.name, section) {
                            cur = Some((t, da.scalar_type, da.num_comp, da.name, da.is_ascii));
                            text_buf.clear();
                        } else {
                            skip_to_end = Some(qname.as_ref().to_vec());
                        }
                    }
                    "AppendedData" => {
                        for attr in e.attributes().flatten() {
                            let key =
                                std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
                            let val = std::str::from_utf8(&attr.value).unwrap_or("");
                            if key == "encoding" && val == "raw" {
                                appended_enc = AppendedEncoding::Raw;
                            }
                        }
                        // Find the `_` sentinel that marks the start of binary data
                        let pos = reader.buffer_position() as usize;
                        if let Some(idx) = raw[pos..].iter().position(|&b| b == b'_') {
                            appended_data_start = Some(pos + idx + 1);
                        }
                        break; // rest is binary, stop XML parsing
                    }
                    _ => {}
                }
            }

            // Self-closing: <DataArray ... format="appended" offset="N"/>
            Ok(Event::Empty(ref e)) => {
                let qname = e.name();
                let tag = std::str::from_utf8(qname.as_ref()).unwrap_or("");
                if tag == "DataArray" {
                    let da = parse_da_attrs(e);
                    let skip = should_skip(&da.name, section, skip_cells, skip_point_data, filter);
                    if da.is_appended && !skip {
                        if let Some(t) = array_target_for(&da.name, section) {
                            appended_entries.push(AppendedEntry {
                                name: da.name,
                                target: t,
                                scalar_type: da.scalar_type,
                                num_comp: da.num_comp,
                                offset: da.appended_offset,
                            });
                        }
                    }
                }
            }

            Ok(Event::End(ref e)) => {
                let qname = e.name();
                let tag = std::str::from_utf8(qname.as_ref()).unwrap_or("");
                match tag {
                    "PointData" | "CellData" | "Points" | "Cells" | "Verts"
                    | "Lines" | "Strips" | "Polys" | "FieldData" => {
                        section = Section::None;
                    }
                    "DataArray" => {
                        if let Some((target, scalar_type, num_comp, name, _is_ascii)) =
                            cur.take()
                        {
                            pending.push(PendingArray {
                                name,
                                target,
                                scalar_type,
                                num_comp,
                                text: std::mem::take(&mut text_buf),
                            });
                        }
                    }
                    _ => {}
                }
            }

            Ok(Event::Text(ref e)) => {
                if cur.is_some() {
                    if let Ok(s) = std::str::from_utf8(e.as_ref()) {
                        text_buf.push_str(s);
                    }
                }
            }

            Ok(Event::Eof) => break,
            Ok(_) => {}

            Err(e) => {
                return Err(VTKParseError::Xml(e));
            }
        }

        // Fast-skip: called after the match so borrows on `buf` are released
        if let Some(tag_name) = skip_to_end.take() {
            reader
                .read_to_end_into(QName(&tag_name), &mut buf)
                .map_err(VTKParseError::Xml)?;
        }

        buf.clear();
    }

    // ------------------------------------------------------------------
    // Phase 2: decode ALL inline arrays in parallel
    // ------------------------------------------------------------------
    let decoded_inline: Vec<Result<(ArrayTarget, String, DecodedArray), String>> = pending
        .into_par_iter()
        .map(|p| {
            let is_ascii = p.text.as_bytes().iter().all(|&b| {
                b.is_ascii_whitespace()
                    || b.is_ascii_digit()
                    || b == b'.'
                    || b == b'-'
                    || b == b'+'
                    || b == b'e'
                    || b == b'E'
            });

            let data = if is_ascii {
                parse_ascii_to_bytes(&p.text, p.scalar_type)
            } else {
                decode_data_array(&p.text, encoding)
                    .map_err(|e| format!("array '{}': {e}", p.name))?
            };

            Ok((
                p.target,
                p.name,
                DecodedArray {
                    data,
                    scalar_type: p.scalar_type,
                    num_comp: p.num_comp,
                },
            ))
        })
        .collect();

    // ------------------------------------------------------------------
    // Phase 3: decode appended-data arrays in parallel (if any)
    // ------------------------------------------------------------------
    let decoded_appended: Vec<Result<(ArrayTarget, String, DecodedArray), String>> =
        if let Some(data_start) = appended_data_start {
            appended_entries
                .into_par_iter()
                .map(|entry| {
                    let data = decode_appended(
                        raw,
                        data_start,
                        entry.offset,
                        encoding,
                        appended_enc,
                    )
                    .map_err(|e| format!("appended array '{}': {e}", entry.name))?;
                    Ok((
                        entry.target,
                        entry.name,
                        DecodedArray {
                            data,
                            scalar_type: entry.scalar_type,
                            num_comp: entry.num_comp,
                        },
                    ))
                })
                .collect()
        } else {
            Vec::new()
        };

    // ------------------------------------------------------------------
    // Phase 4: assemble into MeshArrays
    // ------------------------------------------------------------------
    let mut points: Option<DecodedArray> = Option::None;
    let mut connectivity: Option<DecodedArray> = Option::None;
    let mut offsets: Option<DecodedArray> = Option::None;
    let mut cell_types: Option<DecodedArray> = Option::None;
    let mut point_data: std::collections::HashMap<String, DecodedArray> =
        std::collections::HashMap::new();
    let mut cell_data: std::collections::HashMap<String, DecodedArray> =
        std::collections::HashMap::new();

    for result in decoded_inline
        .into_iter()
        .chain(decoded_appended.into_iter())
    {
        let (target, name, arr) =
            result.map_err(|e| VTKParseError::InvalidFormat(e))?;
        match target {
            ArrayTarget::Points => points = Some(arr),
            ArrayTarget::Connectivity => connectivity = Some(arr),
            ArrayTarget::Offsets => offsets = Some(arr),
            ArrayTarget::CellTypes => cell_types = Some(arr),
            ArrayTarget::PointData => {
                point_data.insert(name, arr);
            }
            ArrayTarget::CellData => {
                cell_data.insert(name, arr);
            }
        }
    }

    let points = points.ok_or_else(|| {
        VTKParseError::InvalidFormat("No Points array found in VTK file".into())
    })?;

    Ok(MeshArrays {
        n_points,
        n_cells,
        points,
        cells: connectivity,
        cell_offsets: offsets,
        cell_types,
        point_data,
        cell_data,
    })
}

// ---------------------------------------------------------------------------
// Inline binary data decoding
// ---------------------------------------------------------------------------

/// Decode the base64-encoded, possibly zlib-compressed content of a DataArray.
fn decode_data_array(text: &str, ei: EncodingInfo) -> Result<Vec<u8>, String> {
    let clean: String = text
        .as_bytes()
        .iter()
        .filter(|&&b| !b.is_ascii_whitespace())
        .map(|&b| b as char)
        .collect();

    if clean.is_empty() {
        return Ok(Vec::new());
    }

    if ei.compressor == Compressor::None {
        // Uncompressed inline binary: [header_size_bytes][raw_data]
        let bytes = B64.decode(&clean).map_err(|e| format!("base64: {e}"))?;
        let hdr = ei.header_bytes();
        if bytes.len() <= hdr {
            return Ok(Vec::new());
        }
        return Ok(bytes[hdr..].to_vec());
    }

    // Compressed inline binary
    decode_compressed_inline(&clean, ei)
}

/// Decode VTK compressed inline binary data.
///
/// VTK writes the compression header and data as independently base64-encoded
/// segments concatenated together.
fn decode_compressed_inline(
    b64_text: &str,
    ei: EncodingInfo,
) -> Result<Vec<u8>, String> {
    let hdr_bytes = ei.header_bytes();

    // Step 1: decode nb (number of blocks)
    let hdr1_b64_len = b64_len(hdr_bytes);
    if b64_text.len() < hdr1_b64_len {
        return Err("base64 too short for header".into());
    }
    let hdr1_raw = B64
        .decode(&b64_text[..hdr1_b64_len])
        .map_err(|e| format!("base64 header(nb): {e}"))?;
    let nb = read_header_val(&hdr1_raw, 0, ei) as usize;

    // Step 2: full header: [nb, nu, np, nc_1..nc_nb]
    let full_hdr_bytes = hdr_bytes * (3 + nb);
    let full_hdr_b64_len = b64_len(full_hdr_bytes);
    if b64_text.len() < full_hdr_b64_len {
        return Err("base64 too short for full header".into());
    }
    let full_hdr_raw = B64
        .decode(&b64_text[..full_hdr_b64_len])
        .map_err(|e| format!("base64 full header: {e}"))?;

    let nu = read_header_val(&full_hdr_raw, hdr_bytes, ei) as usize;
    let np = read_header_val(&full_hdr_raw, 2 * hdr_bytes, ei) as usize;

    let mut compressed_sizes = Vec::with_capacity(nb);
    for i in 0..nb {
        compressed_sizes
            .push(read_header_val(&full_hdr_raw, (3 + i) * hdr_bytes, ei) as usize);
    }
    let total_compressed: usize = compressed_sizes.iter().sum();

    // Step 3: decode the data segment
    let data_b64_start = full_hdr_b64_len;
    let data_b64_len = b64_len(total_compressed);
    let data_b64_end = data_b64_start + data_b64_len;
    if b64_text.len() < data_b64_end {
        return Err(format!(
            "base64 too short for data: need {data_b64_end} have {}",
            b64_text.len()
        ));
    }
    let compressed_data = B64
        .decode(&b64_text[data_b64_start..data_b64_end])
        .map_err(|e| format!("base64 data: {e}"))?;

    decompress_blocks(&compressed_data, &compressed_sizes, nu, np)
}

// ---------------------------------------------------------------------------
// Appended data decoding
// ---------------------------------------------------------------------------

/// Decode a single array from the AppendedData section.
fn decode_appended(
    raw: &[u8],
    data_start: usize,
    offset: usize,
    ei: EncodingInfo,
    appended_enc: AppendedEncoding,
) -> Result<Vec<u8>, String> {
    let array_start = data_start + offset;
    if array_start >= raw.len() {
        return Err(format!(
            "appended offset {offset} out of bounds (data_start={data_start}, file_len={})",
            raw.len()
        ));
    }

    if appended_enc == AppendedEncoding::Raw {
        decode_appended_raw(raw, array_start, ei)
    } else {
        decode_appended_base64(raw, array_start, ei)
    }
}

/// Decode raw-encoded appended data.
fn decode_appended_raw(
    raw: &[u8],
    start: usize,
    ei: EncodingInfo,
) -> Result<Vec<u8>, String> {
    let hdr = ei.header_bytes();
    if start + hdr > raw.len() {
        return Err("appended raw: header out of bounds".into());
    }

    if ei.compressor == Compressor::None {
        let byte_count = read_header_val(raw, start, ei) as usize;
        let data_off = start + hdr;
        let data_end = data_off + byte_count;
        if data_end > raw.len() {
            return Err("appended raw: data out of bounds".into());
        }
        Ok(raw[data_off..data_end].to_vec())
    } else {
        decode_compressed_appended_raw(raw, start, ei)
    }
}

/// Decode compressed raw appended data (zlib blocks).
fn decode_compressed_appended_raw(
    raw: &[u8],
    start: usize,
    ei: EncodingInfo,
) -> Result<Vec<u8>, String> {
    let hdr = ei.header_bytes();
    let nb = read_header_val(raw, start, ei) as usize;
    let full_hdr = hdr * (3 + nb);
    if start + full_hdr > raw.len() {
        return Err("compressed appended raw: header out of bounds".into());
    }

    let nu = read_header_val(raw, start + hdr, ei) as usize;
    let np = read_header_val(raw, start + 2 * hdr, ei) as usize;

    let mut compressed_sizes = Vec::with_capacity(nb);
    for i in 0..nb {
        compressed_sizes
            .push(read_header_val(raw, start + (3 + i) * hdr, ei) as usize);
    }
    let total_compressed: usize = compressed_sizes.iter().sum();

    let data_off = start + full_hdr;
    let data_end = data_off + total_compressed;
    if data_end > raw.len() {
        return Err("compressed appended raw: data out of bounds".into());
    }

    decompress_blocks(&raw[data_off..data_end], &compressed_sizes, nu, np)
}

/// Decode base64-encoded appended data (fallback).
fn decode_appended_base64(
    raw: &[u8],
    start: usize,
    ei: EncodingInfo,
) -> Result<Vec<u8>, String> {
    // Find the end of the base64 segment (next `<` or EOF)
    let end = raw[start..]
        .iter()
        .position(|&b| b == b'<')
        .map(|i| start + i)
        .unwrap_or(raw.len());

    let b64_text = std::str::from_utf8(&raw[start..end])
        .map_err(|e| format!("appended base64 utf8: {e}"))?;
    let clean: String = b64_text
        .as_bytes()
        .iter()
        .filter(|&&b| !b.is_ascii_whitespace())
        .map(|&b| b as char)
        .collect();

    if clean.is_empty() {
        return Ok(Vec::new());
    }

    if ei.compressor == Compressor::None {
        let bytes = B64.decode(&clean).map_err(|e| format!("base64: {e}"))?;
        let hdr = ei.header_bytes();
        if bytes.len() <= hdr {
            return Ok(Vec::new());
        }
        Ok(bytes[hdr..].to_vec())
    } else {
        decode_compressed_inline(&clean, ei)
    }
}

// ---------------------------------------------------------------------------
// Shared zlib decompression (parallel via rayon)
// ---------------------------------------------------------------------------

fn decompress_blocks(
    compressed_data: &[u8],
    compressed_sizes: &[usize],
    nu: usize,
    np: usize,
) -> Result<Vec<u8>, String> {
    let nb = compressed_sizes.len();
    let mut block_starts = Vec::with_capacity(nb + 1);
    block_starts.push(0usize);
    for &nc in compressed_sizes {
        block_starts.push(block_starts.last().unwrap() + nc);
    }

    let last_block_size = if np > 0 { np } else { nu };

    let blocks: Vec<Result<Vec<u8>, String>> = (0..nb)
        .into_par_iter()
        .map(|i| {
            let c_start = block_starts[i];
            let c_end = block_starts[i + 1];
            if c_end > compressed_data.len() {
                return Err(format!(
                    "block {i}: range {c_start}..{c_end} exceeds data len {}",
                    compressed_data.len()
                ));
            }
            let expected = if i < nb - 1 { nu } else { last_block_size };
            let mut decoder =
                flate2::read::ZlibDecoder::new(&compressed_data[c_start..c_end]);
            let mut out = Vec::with_capacity(expected);
            decoder
                .read_to_end(&mut out)
                .map_err(|e| format!("zlib block {i}: {e}"))?;
            Ok(out)
        })
        .collect();

    let total = if nb > 0 {
        nu * (nb - 1) + last_block_size
    } else {
        0
    };
    let mut output = Vec::with_capacity(total);
    for r in blocks {
        output.extend_from_slice(&r?);
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Number of base64 characters needed to encode `n` bytes.
fn b64_len(n: usize) -> usize {
    (n + 2) / 3 * 4
}

/// Read a single header value (u32 or u64 depending on header_type).
fn read_header_val(data: &[u8], offset: usize, ei: EncodingInfo) -> u64 {
    match ei.header_type {
        ScalarType::UInt64 => {
            if offset + 8 <= data.len() {
                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
            } else {
                0
            }
        }
        _ => {
            if offset + 4 <= data.len() {
                u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as u64
            } else {
                0
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn no_filter() -> ArrayFilter {
        ArrayFilter::new(None, None)
    }

    #[test]
    fn test_parse_simple_vtu_ascii() {
        let xml = br#"<?xml version="1.0"?>
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

        let mesh = parse_vtk_xml(xml, &no_filter(), false, false).unwrap();
        assert_eq!(mesh.n_points, 4);
        assert_eq!(mesh.n_cells, 1);
        // Points: 4 * 3 * 8 bytes = 96 bytes
        assert_eq!(mesh.points.data.len(), 96);
        assert_eq!(mesh.points.scalar_type, ScalarType::Float64);
        // Connectivity: 4 * 8 bytes = 32
        let cells = mesh.cells.unwrap();
        assert_eq!(cells.data.len(), 32);
        // Offsets: 1 * 8 bytes = 8
        let offsets = mesh.cell_offsets.unwrap();
        assert_eq!(offsets.data.len(), 8);
        // Types: 1 byte
        let types = mesh.cell_types.unwrap();
        assert_eq!(types.data.len(), 1);
        assert_eq!(types.data[0], 9);
    }

    #[test]
    fn test_parse_vtu_with_point_data() {
        let xml = br#"<?xml version="1.0"?>
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

        let mesh = parse_vtk_xml(xml, &no_filter(), false, false).unwrap();
        assert_eq!(mesh.n_points, 3);
        assert!(mesh.point_data.contains_key("Temperature"));
        let temp = &mesh.point_data["Temperature"];
        assert_eq!(temp.num_comp, 1);
        // 3 floats * 8 bytes = 24
        assert_eq!(temp.data.len(), 24);
    }

    #[test]
    fn test_parse_vtp_polydata() {
        let xml = br#"<?xml version="1.0"?>
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

        let mesh = parse_vtk_xml(xml, &no_filter(), false, false).unwrap();
        assert_eq!(mesh.n_points, 3);
        assert_eq!(mesh.n_cells, 1);
    }

    #[test]
    fn test_skip_cells() {
        let xml = br#"<?xml version="1.0"?>
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
        <DataArray Name="T" type="Float64" NumberOfComponents="1" format="ascii">
          1.0 2.0 3.0
        </DataArray>
      </PointData>
      <CellData>
        <DataArray Name="V" type="Float64" NumberOfComponents="1" format="ascii">
          10.0
        </DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#;

        let mesh = parse_vtk_xml(xml, &no_filter(), true, false).unwrap();
        assert_eq!(mesh.n_points, 3);
        assert!(mesh.cells.is_none());
        assert!(mesh.cell_offsets.is_none());
        assert!(mesh.cell_types.is_none());
        assert!(mesh.cell_data.is_empty());
        assert!(mesh.point_data.contains_key("T"));
    }

    #[test]
    fn test_include_filter() {
        let xml = br#"<?xml version="1.0"?>
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
          1.0 2.0 3.0
        </DataArray>
        <DataArray Name="Pressure" type="Float64" NumberOfComponents="1" format="ascii">
          10.0 20.0 30.0
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#;

        let filter = ArrayFilter::new(Some(vec!["Temperature".to_string()]), None);
        let mesh = parse_vtk_xml(xml, &filter, false, false).unwrap();
        assert!(mesh.point_data.contains_key("Temperature"));
        assert!(!mesh.point_data.contains_key("Pressure"));
    }

    #[test]
    fn test_inline_binary_uncompressed() {
        // Create a minimal VTU with inline binary (base64) data.
        // 3 points, Float64, 3 components = 9 * 8 = 72 bytes of data.
        // Header: UInt32 = 4 bytes with value 72.
        use base64::engine::general_purpose::STANDARD as B64;
        use base64::Engine;

        let point_vals: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0];
        let mut raw_bytes: Vec<u8> = Vec::new();
        // header: UInt32 little-endian = 72
        raw_bytes.extend_from_slice(&72u32.to_le_bytes());
        for &v in &point_vals {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let b64_str = B64.encode(&raw_bytes);

        let xml = format!(
            r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="0">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="binary">
          {b64_str}
        </DataArray>
      </Points>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#
        );

        let mesh = parse_vtk_xml(xml.as_bytes(), &no_filter(), false, false).unwrap();
        assert_eq!(mesh.n_points, 3);
        // Verify point data: 9 * 8 = 72 bytes
        assert_eq!(mesh.points.data.len(), 72);
        // First point should be (0, 0, 0)
        let first_x = f64::from_le_bytes(mesh.points.data[0..8].try_into().unwrap());
        assert!((first_x - 0.0).abs() < 1e-10);
        // Second point x should be 1.0
        let second_x = f64::from_le_bytes(mesh.points.data[24..32].try_into().unwrap());
        assert!((second_x - 1.0).abs() < 1e-10);
    }
}
