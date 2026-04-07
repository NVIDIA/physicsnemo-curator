// SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ASE LMDB reader module.
//!
//! Reads `.aselmdb` files produced by the
//! [ase-db-backends](https://github.com/NVIDIA/ase-db-backends) package.
//! Each value is zlib-compressed JSON using ASE's custom encoding for
//! NumPy arrays (`__ndarray__` markers).
//!
//! This module provides a fast Rust-based reader that bypasses the
//! Python ASE deserialization stack, returning raw row dictionaries
//! with NumPy arrays directly to Python.

pub mod bindings;
pub mod reader;
