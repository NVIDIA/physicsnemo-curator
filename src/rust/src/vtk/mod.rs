// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! VTK file format reader module.
//!
//! Provides multi-threaded reading of VTK XML files (VTU, VTP) with
//! full support for ASCII, inline binary (base64 ± zlib), and appended
//! data (raw or base64 ± zlib). Converts to native-dtype NumPy arrays
//! for Python integration.

pub mod bindings;
pub mod mesh;
pub mod parser;
pub mod reader;
