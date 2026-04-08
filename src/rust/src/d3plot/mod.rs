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

//! LS-DYNA d3plot post-processing utilities.
//!
//! Provides Rust-accelerated implementations of three compute-intensive
//! operations used when reading crash simulation meshes:
//!
//! - **K-file parsing** — extracts part→thickness mappings from `.k` files
//! - **Node thickness** — scatter-accumulates element thickness onto nodes
//! - **Von Mises stress** — computes von Mises from Voigt-notation tensors

pub mod bindings;
pub mod kfile;
pub mod stress;
pub mod thickness;
