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

//! Per-node thickness computation from element connectivity.
//!
//! Scatter-accumulates element-level thickness values onto mesh nodes
//! using the element connectivity table, then averages by node valence.

use std::collections::HashMap;

/// Compute per-node thickness averaged from incident elements.
///
/// For each element, its thickness is looked up from the
/// `part_thickness_map` via the `part_ids` array.  The thickness is
/// then scattered onto each node in the element's connectivity row.
/// Finally, each node's accumulated thickness is divided by its
/// incident element count.
///
/// # Arguments
///
/// * `connectivity` — `(E, nodes_per_cell)` flattened row-major.
/// * `n_elements` — number of elements E.
/// * `nodes_per_cell` — number of nodes per element (e.g., 4 for quads).
/// * `part_ids` — `(E,)` part index per element.
/// * `part_thickness_map` — actual part ID → thickness.
/// * `actual_part_ids` — optional array mapping part-index → part-ID.
///   If `None`, a sorted enumeration of `part_thickness_map` keys is used.
///
/// # Returns
///
/// Per-node thickness array of length `max_node + 1`.
pub fn compute_node_thickness(
    connectivity: &[i64],
    n_elements: usize,
    nodes_per_cell: usize,
    part_ids: &[i64],
    part_thickness_map: &HashMap<i64, f64>,
    actual_part_ids: Option<&[i64]>,
) -> Vec<f64> {
    // Build part-index → part-ID mapping.
    if n_elements == 0 {
        return Vec::new();
    }

    let part_index_to_id: HashMap<i64, i64> = match actual_part_ids {
        Some(ids) => ids
            .iter()
            .enumerate()
            .filter(|&(i, _)| i > 0)
            .map(|(i, &pid)| (i as i64, pid))
            .collect(),
        None => {
            let mut sorted_pids: Vec<i64> = part_thickness_map.keys().copied().collect();
            sorted_pids.sort();
            sorted_pids
                .into_iter()
                .enumerate()
                .map(|(i, pid)| ((i + 1) as i64, pid))
                .collect()
        }
    };

    // Map each element to its thickness value.
    let mut element_thickness = vec![0.0_f64; n_elements];
    for (i, &part_index) in part_ids.iter().enumerate() {
        if let Some(&actual_id) = part_index_to_id.get(&part_index) {
            element_thickness[i] = part_thickness_map.get(&actual_id).copied().unwrap_or(0.0);
        }
    }

    // Find max node index to size the output array.
    let max_node = connectivity.iter().copied().max().unwrap_or(0) as usize;
    let n_nodes = max_node + 1;
    let mut node_thickness = vec![0.0_f64; n_nodes];
    let mut node_count = vec![0.0_f64; n_nodes];

    // Scatter-accumulate element thickness onto nodes.
    for (elem_idx, &t) in element_thickness.iter().enumerate() {
        let row_start = elem_idx * nodes_per_cell;
        for col in 0..nodes_per_cell {
            let node_idx = connectivity[row_start + col] as usize;
            node_thickness[node_idx] += t;
            node_count[node_idx] += 1.0;
        }
    }

    // Average by incident count.
    for i in 0..n_nodes {
        if node_count[i] > 0.0 {
            node_thickness[i] /= node_count[i];
        }
    }

    node_thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_quad_element() {
        // 1 element, 4 nodes, part_id=1 → thickness 2.5
        let connectivity = vec![0_i64, 1, 2, 3];
        let part_ids = vec![1_i64];
        let mut pt_map = HashMap::new();
        pt_map.insert(1_i64, 2.5_f64);
        let actual_pids = vec![0_i64, 1]; // index 0 unused, index 1 = part 1

        let result =
            compute_node_thickness(&connectivity, 1, 4, &part_ids, &pt_map, Some(&actual_pids));

        assert_eq!(result.len(), 4);
        for &v in &result {
            assert!((v - 2.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_two_elements_shared_nodes() {
        // 2 elements sharing nodes 1,2.
        // Element 0: [0,1,2,3] part 1 → thickness 2.0
        // Element 1: [1,2,4,5] part 2 → thickness 4.0
        // Nodes 1,2 should average to (2+4)/2 = 3.0
        let connectivity = vec![0, 1, 2, 3, 1, 2, 4, 5];
        let part_ids = vec![1_i64, 2];
        let mut pt_map = HashMap::new();
        pt_map.insert(1_i64, 2.0);
        pt_map.insert(2_i64, 4.0);
        let actual_pids = vec![0_i64, 1, 2];

        let result =
            compute_node_thickness(&connectivity, 2, 4, &part_ids, &pt_map, Some(&actual_pids));

        assert!((result[0] - 2.0).abs() < 1e-10); // only elem 0
        assert!((result[1] - 3.0).abs() < 1e-10); // shared
        assert!((result[2] - 3.0).abs() < 1e-10); // shared
        assert!((result[3] - 2.0).abs() < 1e-10); // only elem 0
        assert!((result[4] - 4.0).abs() < 1e-10); // only elem 1
        assert!((result[5] - 4.0).abs() < 1e-10); // only elem 1
    }

    #[test]
    fn test_no_actual_part_ids() {
        // Without actual_part_ids, sorted enumeration is used.
        // part_thickness_map keys sorted: [10, 20] → index 1→10, index 2→20
        let connectivity = vec![0, 1, 2, 3];
        let part_ids = vec![1_i64]; // index 1 → part 10
        let mut pt_map = HashMap::new();
        pt_map.insert(10_i64, 5.0);
        pt_map.insert(20_i64, 7.0);

        let result = compute_node_thickness(&connectivity, 1, 4, &part_ids, &pt_map, None);
        for &v in &result[..4] {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_empty_connectivity() {
        let connectivity: Vec<i64> = vec![];
        let part_ids: Vec<i64> = vec![];
        let pt_map = HashMap::new();

        let result = compute_node_thickness(&connectivity, 0, 4, &part_ids, &pt_map, None);
        assert!(result.is_empty());
    }
}
