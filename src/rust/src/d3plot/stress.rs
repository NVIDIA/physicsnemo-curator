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

//! Von Mises stress computation from Voigt-notation stress tensors.
//!
//! Voigt ordering: `[sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_zx]`.

/// Compute von Mises stress from a flattened Voigt stress array.
///
/// The input `stress` is a contiguous row-major array of shape
/// `(..., 6)` where the last dimension holds the 6 Voigt components.
/// `n_total` is the total number of elements (product of all leading
/// dimensions), i.e. `stress.len() == n_total * 6`.
///
/// Returns a `Vec<f64>` of length `n_total` with the scalar von Mises
/// stress for each entry.
pub fn von_mises_from_voigt(stress: &[f64], n_total: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; n_total];

    for (i, chunk) in stress.chunks_exact(6).enumerate().take(n_total) {
        let sx = chunk[0];
        let sy = chunk[1];
        let sz = chunk[2];
        let txy = chunk[3];
        let tyz = chunk[4];
        let tzx = chunk[5];

        let j2 = 0.5 * ((sx - sy).powi(2) + (sy - sz).powi(2) + (sz - sx).powi(2))
            + 3.0 * (txy.powi(2) + tyz.powi(2) + tzx.powi(2));

        result[i] = j2.max(0.0).sqrt();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniaxial_stress() {
        // sigma_x = 100, rest = 0 → von Mises = 100
        let stress = vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = von_mises_from_voigt(&stress, 1);
        assert!((result[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_pure_shear() {
        // tau_xy = 50, rest = 0 → von Mises = sqrt(3) * 50 ≈ 86.60
        let stress = vec![0.0, 0.0, 0.0, 50.0, 0.0, 0.0];
        let result = von_mises_from_voigt(&stress, 1);
        let expected = (3.0_f64).sqrt() * 50.0;
        assert!((result[0] - expected).abs() < 1e-8);
    }

    #[test]
    fn test_hydrostatic_stress() {
        // sigma_x = sigma_y = sigma_z = p, shear = 0 → von Mises = 0
        let p = 42.0;
        let stress = vec![p, p, p, 0.0, 0.0, 0.0];
        let result = von_mises_from_voigt(&stress, 1);
        assert!(result[0].abs() < 1e-10);
    }

    #[test]
    fn test_batch() {
        // Two entries: uniaxial 100 and uniaxial 200
        let stress = vec![
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let result = von_mises_from_voigt(&stress, 2);
        assert!((result[0] - 100.0).abs() < 1e-10);
        assert!((result[1] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_stress() {
        let stress = vec![0.0; 6];
        let result = von_mises_from_voigt(&stress, 1);
        assert!(result[0].abs() < 1e-15);
    }
}
