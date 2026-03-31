# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build script for ASV benchmark environments.

ASV calls this after checking out a commit to build the Rust extension
inside the isolated virtualenv. It ensures the Rust toolchain is on PATH
before invoking maturin.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_rust_bin() -> str | None:
    """Locate the directory containing ``rustc``."""
    # Already on PATH?
    if shutil.which("rustc"):
        return None  # no modification needed

    home = Path.home()
    candidates = [
        home / ".cargo" / "bin",
        home / ".rustup" / "toolchains" / "stable-x86_64-unknown-linux-gnu" / "bin",
        Path("/usr/local/cargo/bin"),
    ]
    for candidate in candidates:
        if (candidate / "rustc").is_file():
            return str(candidate)

    return None


def main() -> None:
    """Install maturin and build the native extension."""
    # Ensure Rust is reachable
    rust_bin = _find_rust_bin()
    env = os.environ.copy()
    if rust_bin:
        env["PATH"] = rust_bin + os.pathsep + env.get("PATH", "")

    if not shutil.which("rustc", path=env.get("PATH")):
        print("ERROR: rustc not found. Install Rust via https://rustup.rs/", file=sys.stderr)
        sys.exit(1)

    # Install maturin into the ASV virtualenv
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "maturin>=1.0,<2.0"],
        env=env,
    )

    # Build native extension in release mode
    subprocess.check_call(
        ["maturin", "develop", "--release", "--manifest-path", "src/rust/Cargo.toml"],
        env=env,
    )


if __name__ == "__main__":
    main()
