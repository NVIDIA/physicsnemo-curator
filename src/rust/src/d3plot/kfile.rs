// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Parser for LS-DYNA `.k` keyword files.
//!
//! Extracts `*PART` definitions (part ID → section ID) and
//! `*SECTION_SHELL` definitions (section ID → thickness) to produce
//! a `HashMap<i64, f64>` mapping part ID → average thickness.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Parse an LS-DYNA `.k` keyword file for part thickness.
///
/// Returns a mapping from part ID to thickness value.  The logic
/// mirrors the Python `_parse_k_file` exactly:
///
/// 1. Strip blank lines and `$` comment lines.
/// 2. Scan for `*PART` keywords → extract `(part_id, section_id)`.
/// 3. Scan for `*SECTION_SHELL` keywords → extract `(section_id, thickness)`.
/// 4. Join on section_id to produce `part_id → thickness`.
pub fn parse_k_file(path: &Path) -> Result<HashMap<i64, f64>, std::io::Error> {
    let content = fs::read_to_string(path)?;

    // Pre-filter: strip blank and comment lines.
    let lines: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('$'))
        .collect();

    let mut part_to_section: HashMap<i64, i64> = HashMap::new();
    let mut section_thickness: HashMap<i64, f64> = HashMap::new();

    let mut i = 0;
    while i < lines.len() {
        let line_upper = lines[i].to_uppercase();

        if line_upper.contains("*PART") && !line_upper.contains("*PART_") {
            // i+1 = part name (skip), i+2 = part_id section_id material_id ...
            if i + 2 < lines.len() {
                let tokens: Vec<&str> = lines[i + 2].split_whitespace().collect();
                if tokens.len() >= 2 {
                    if let (Ok(part_id), Ok(section_id)) =
                        (tokens[0].parse::<i64>(), tokens[1].parse::<i64>())
                    {
                        part_to_section.insert(part_id, section_id);
                    }
                }
            }
            i += 3;
        } else if line_upper.contains("*SECTION_SHELL") {
            i += 1; // skip keyword line
            while i < lines.len() && !lines[i].starts_with('*') {
                let first_char = lines[i].chars().next().unwrap_or(' ');
                if first_char.is_ascii_digit() {
                    // Header line: section_id ...
                    let header_tokens: Vec<&str> = lines[i].split_whitespace().collect();
                    let section_id = header_tokens.first().and_then(|t| t.parse::<i64>().ok());

                    // Thickness line follows.
                    let thickness_line = if i + 1 < lines.len() {
                        lines[i + 1]
                    } else {
                        ""
                    };

                    let thickness_values: Vec<f64> = thickness_line
                        .split_whitespace()
                        .map(|tok| tok.parse::<f64>().unwrap_or(0.0))
                        .collect();

                    let non_zero: Vec<f64> = thickness_values
                        .iter()
                        .copied()
                        .filter(|&t| t > 0.0)
                        .collect();

                    let thickness = if !non_zero.is_empty() {
                        non_zero.iter().sum::<f64>() / non_zero.len() as f64
                    } else if !thickness_values.is_empty() {
                        thickness_values.iter().sum::<f64>() / thickness_values.len() as f64
                    } else {
                        0.0
                    };

                    if let Some(sid) = section_id {
                        section_thickness.insert(sid, thickness);
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
    }

    // Join: part_id → thickness via section_id.
    let result: HashMap<i64, f64> = part_to_section
        .iter()
        .map(|(&pid, &sid)| (pid, section_thickness.get(&sid).copied().unwrap_or(0.0)))
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_single_part() {
        let content = r#"$
*KEYWORD
*PART
Part_1
       1       1       1
*SECTION_SHELL
       1
     2.5     2.5     2.5     2.5
*END
"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        let result = parse_k_file(f.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[&1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_two_parts() {
        let content = r#"$
*KEYWORD
*PART
Part_1
       1       1       1
*PART
Part_2
       2       2       2
*SECTION_SHELL
       1
     2.0     2.0     2.0     2.0
       2
     3.0     3.0     3.0     3.0
*END
"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        let result = parse_k_file(f.path()).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[&1] - 2.0).abs() < 1e-10);
        assert!((result[&2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_empty_file() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"").unwrap();
        let result = parse_k_file(f.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_nonexistent_file() {
        let result = parse_k_file(Path::new("/nonexistent/path/file.k"));
        assert!(result.is_err());
    }
}
