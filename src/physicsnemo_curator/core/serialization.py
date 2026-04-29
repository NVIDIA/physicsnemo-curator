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

"""Pipeline serialization and deserialization to YAML and JSON formats.

Provides functions to serialize a :class:`~physicsnemo_curator.core.base.Pipeline`
to a dictionary, save it to disk (YAML or JSON), and reconstruct a live pipeline
from saved configuration.  Class resolution uses :mod:`importlib` and parameter
coercion relies on :meth:`Param.type` declarations from each component's
``params()`` classmethod.

Examples
--------
>>> from physicsnemo_curator.core.serialization import save_pipeline, load_pipeline
>>> save_pipeline(pipeline, "my_pipeline.yaml")        # doctest: +SKIP
>>> restored = load_pipeline("my_pipeline.yaml")       # doctest: +SKIP
>>> restored[0]                                        # doctest: +SKIP
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import pathlib
import platform
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from physicsnemo_curator.core.base import Pipeline
from physicsnemo_curator.core.pipeline_store import _pipeline_config

if TYPE_CHECKING:
    from physicsnemo_curator.core.base import Filter, Sink, Source

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
"""Current serialization format version."""


def _collect_metadata() -> dict[str, Any]:
    """Collect environment metadata for pipeline provenance.

    Gathers package version, Python/platform info, timestamp, and git
    state.  All fields are best-effort — git fields are ``None`` when
    not in a repository or when ``git`` is unavailable.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary with keys: ``psnc_version``,
        ``rust_extension``, ``python_version``, ``platform``,
        ``created_utc``, ``git_hash``, ``git_dirty``.
    """
    from physicsnemo_curator import __version__, rust_version

    # Git information (best-effort)
    git_hash: str | None = None
    git_dirty: bool | None = None
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()

            dirty_result = subprocess.run(  # noqa: S603, S607
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if dirty_result.returncode == 0:
                git_dirty = len(dirty_result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("git not available; skipping git metadata")

    return {
        "psnc_version": __version__,
        "rust_extension": rust_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
    }


def serialize_pipeline(pipeline: Pipeline[Any]) -> dict[str, Any]:
    """Serialize a pipeline to a configuration dictionary.

    Calls :func:`~physicsnemo_curator.core.pipeline_store._pipeline_config`
    and adds ``version`` and ``metadata`` keys.  Ensures the ``sink`` key
    is always present (set to ``None`` when the pipeline has no sink).

    The ``metadata`` section captures provenance information (package
    version, git hash, timestamp, platform) and is purely informational —
    it is ignored on deserialization.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to serialize.

    Returns
    -------
    dict[str, Any]
        Serialized pipeline configuration with keys ``version``,
        ``metadata``, ``source``, ``filters``, and ``sink``.
    """
    config = _pipeline_config(pipeline)
    config["version"] = _FORMAT_VERSION
    config["metadata"] = _collect_metadata()
    config.setdefault("sink", None)
    return config


def save_pipeline(pipeline: Pipeline[Any], path: str | pathlib.Path) -> None:
    """Serialize a pipeline and write it to a YAML or JSON file.

    The format is determined by the file extension: ``.yaml`` / ``.yml`` for
    YAML, ``.json`` for JSON.  Parent directories are created automatically.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to save.
    path : str | pathlib.Path
        Destination file path.

    Raises
    ------
    ValueError
        If the file extension is not ``.yaml``, ``.yml``, or ``.json``.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = serialize_pipeline(pipeline)
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            msg = "PyYAML is required for YAML serialization. Install it with: pip install pyyaml"
            raise ImportError(msg) from exc

        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    elif suffix == ".json":
        path.write_text(json.dumps(data, indent=2, default=str))

    else:
        msg = f"Unsupported file extension {suffix!r}. Use .yaml, .yml, or .json."
        raise ValueError(msg)


def _resolve_class(class_name: str, module_path: str) -> type:
    """Import a module and return the named class.

    Parameters
    ----------
    class_name : str
        Name of the class to resolve.
    module_path : str
        Fully qualified module path.

    Returns
    -------
    type
        The resolved class object.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the class does not exist in the module.
    """
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _coerce_params(cls: type, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Coerce serialized parameter values to their declared types.

    Uses the component's ``params()`` classmethod for type declarations and
    ``inspect.signature(cls.__init__)`` to filter to valid ``__init__`` params.
    Handles bool coercion from strings and skips the ``"<REQUIRED>"`` sentinel.

    Parameters
    ----------
    cls : type
        The component class with a ``params()`` classmethod.
    raw_params : dict[str, Any]
        Raw parameter dict from serialized config.

    Returns
    -------
    dict[str, Any]
        Coerced parameters ready for ``cls(**params)``.
    """
    # Build a type lookup from the class's declared params
    type_lookup: dict[str, type] = {}
    if hasattr(cls, "params") and callable(cls.params):
        for p in cls.params():  # ty: ignore[call-top-callable, not-iterable]
            type_lookup[p.name] = p.type

    # Determine which params __init__ actually accepts
    sig = inspect.signature(cls.__init__)
    init_params = {name for name, param in sig.parameters.items() if name != "self"}

    coerced: dict[str, Any] = {}
    for key, value in raw_params.items():
        # Skip the REQUIRED sentinel
        if value == "<REQUIRED>":
            continue

        # Skip params not in __init__ signature
        if key not in init_params:
            continue

        # Coerce to declared type if available
        if key in type_lookup and value is not None:
            target_type = type_lookup[key]
            if target_type is bool and isinstance(value, str):
                coerced[key] = value.lower() in {"true", "1", "yes"}
            elif not isinstance(value, target_type):
                try:
                    coerced[key] = target_type(value)
                except (TypeError, ValueError):
                    coerced[key] = value
            else:
                coerced[key] = value
        else:
            coerced[key] = value

    return coerced


def _reconstruct_component(config: dict[str, Any]) -> Source[Any] | Filter[Any] | Sink[Any]:
    """Reconstruct a pipeline component from its serialized config.

    Parameters
    ----------
    config : dict[str, Any]
        Component configuration with ``class``, ``module``, and ``params`` keys.

    Returns
    -------
    Source | Filter | Sink
        The reconstructed component instance.
    """
    cls = _resolve_class(config["class"], config["module"])
    params = _coerce_params(cls, config.get("params", {}))
    return cls(**params)


def deserialize_pipeline(data: dict[str, Any]) -> Pipeline[Any]:
    """Reconstruct a pipeline from a serialized configuration dictionary.

    Parameters
    ----------
    data : dict[str, Any]
        Serialized pipeline config (as produced by :func:`serialize_pipeline`).

    Returns
    -------
    Pipeline
        A fully reconstructed pipeline with source, filters, and optional sink.
    """
    source = _reconstruct_component(data["source"])
    filters = [_reconstruct_component(f) for f in data.get("filters", [])]
    sink = None
    if data.get("sink") is not None:
        sink = _reconstruct_component(data["sink"])

    return Pipeline(source=source, filters=filters, sink=sink)  # ty: ignore[invalid-argument-type]


def load_pipeline(path: str | pathlib.Path) -> Pipeline[Any]:
    """Load a pipeline from a YAML or JSON file.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the serialized pipeline file.

    Returns
    -------
    Pipeline
        The deserialized pipeline.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = pathlib.Path(path)
    if not path.exists():
        msg = f"Pipeline file not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    text = path.read_text()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            msg = "PyYAML is required for YAML deserialization. Install it with: pip install pyyaml"
            raise ImportError(msg) from exc

        data = yaml.safe_load(text)

    elif suffix == ".json":
        data = json.loads(text)

    else:
        msg = f"Unsupported file extension {suffix!r}. Use .yaml, .yml, or .json."
        raise ValueError(msg)

    return deserialize_pipeline(data)
