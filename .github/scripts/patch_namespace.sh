#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Patch nvidia-physicsnemo's __init__.py to support namespace package
# discovery so physicsnemo.curator can be found from our editable src/ tree.
#
# nvidia-physicsnemo installs a regular physicsnemo/__init__.py which prevents
# Python from merging physicsnemo sub-packages across multiple sys.path entries.
# Adding pkgutil.extend_path makes it a cooperative namespace package.

set -euo pipefail

INIT=$(uv run python -c "
try:
    import physicsnemo
    print(physicsnemo.__file__)
except Exception:
    pass
" 2>/dev/null || true)

if [ -n "$INIT" ] && [ -f "$INIT" ]; then
    if ! grep -q "extend_path" "$INIT"; then
        echo "" >> "$INIT"
        echo "# Auto-patched: extend namespace for physicsnemo-curator" >> "$INIT"
        echo "import pkgutil; __path__ = pkgutil.extend_path(__path__, __name__)" >> "$INIT"
        echo "Patched $INIT with pkgutil.extend_path"
    else
        echo "Already patched: $INIT"
    fi
else
    echo "physicsnemo not installed as regular package — no patch needed"
fi
