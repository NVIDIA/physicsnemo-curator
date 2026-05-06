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

"""Unit tests for partition intersection and batching logic."""

from __future__ import annotations

import pytest

from physicsnemo_curator.run.base import batch_groups, intersect_partitions

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# intersect_partitions
# ---------------------------------------------------------------------------


class TestIntersectPartitions:
    """Tests for intersect_partitions logic."""

    def test_both_none_returns_none(self):
        """If both source and sink have no constraints, return None."""
        assert intersect_partitions(None, None) is None

    def test_source_only(self):
        """If only source has constraints, return source groups."""
        groups = [[0, 1, 2], [3, 4, 5]]
        result = intersect_partitions(groups, None)
        assert result == groups

    def test_sink_only(self):
        """If only sink has constraints, return sink groups."""
        groups = [[0, 1], [2, 3], [4, 5]]
        result = intersect_partitions(None, groups)
        assert result == groups

    def test_identical_partitions(self):
        """If source and sink have identical groups, return them unchanged."""
        groups = [[0, 1, 2], [3, 4, 5]]
        result = intersect_partitions(groups, groups)
        assert result == groups

    def test_compatible_refinement(self):
        """Source groups are identical to sink groups — no conflict."""
        # Source and sink have the same partition.
        source = [[0, 1], [2, 3], [4, 5]]
        sink = [[0, 1], [2, 3], [4, 5]]
        result = intersect_partitions(source, sink)
        assert result == [[0, 1], [2, 3], [4, 5]]

    def test_sink_subset_of_source(self):
        """Sink groups are exact subsets of source groups — source is split, raises."""
        # Source: coarser (requires 0,1,2,3 together)
        source = [[0, 1, 2, 3], [4, 5, 6, 7]]
        # Sink: finer (splits source groups)
        sink = [[0, 1], [2, 3], [4, 5], [6, 7]]
        # Source group [0,1,2,3] is split by sink → incompatible.
        with pytest.raises(ValueError, match="source requires indices"):
            intersect_partitions(source, sink)

    def test_incompatible_source_split_raises(self):
        """Source requires [3,4,5] together but sink splits them."""
        # Source: file requires 3,4,5 together
        source = [[0, 1, 2], [3, 4, 5]]
        # Sink: chunk requires 2,3 together and 4,5 together
        sink = [[0, 1, 2, 3], [4, 5]]
        # Source group [3,4,5] gets split: 3→sink0, 4,5→sink1
        with pytest.raises(ValueError, match="source requires indices"):
            intersect_partitions(source, sink)

    def test_incompatible_sink_split_raises(self):
        """Sink requires indices together but source separates them."""
        # Source groups each map to one sink group, but one sink group
        # draws from multiple source groups — sink group is split.
        source = [[0, 1], [2, 3], [4, 5]]
        sink = [[0, 1, 2, 3], [4, 5]]
        # Source check passes: [0,1]→sink0, [2,3]→sink0, [4,5]→sink1.
        # Sink group [0,1,2,3] spans source0 and source1 — split!
        with pytest.raises(ValueError, match="sink requires indices"):
            intersect_partitions(source, sink)

    def test_mutual_incompatibility_raises(self):
        """Both source and sink have conflicting requirements."""
        source = [[0, 1, 2], [3, 4, 5]]
        # Sink group [1,2,3] spans both source groups
        sink = [[0], [1, 2, 3], [4, 5]]
        with pytest.raises(ValueError):
            intersect_partitions(source, sink)

    def test_single_element_groups_conflict(self):
        """Singleton source groups conflict with multi-element sink groups."""
        # Source says each index is in a different file (must be alone).
        source = [[0], [1], [2], [3]]
        # Sink says 0,1 must be together — incompatible with source.
        sink = [[0, 1], [2, 3]]
        with pytest.raises(ValueError, match="sink requires indices"):
            intersect_partitions(source, sink)

    def test_single_element_groups_compatible(self):
        """Singleton source groups compatible with singleton sink groups."""
        source = [[0], [1], [2], [3]]
        sink = [[0], [1], [2], [3]]
        result = intersect_partitions(source, sink)
        assert result == [[0], [1], [2], [3]]

    def test_result_sorted_by_min_index(self):
        """Intersection groups are sorted by minimum index."""
        # Use both source and sink to trigger intersection path.
        source = [[5, 6], [0, 1], [3, 4]]
        sink = [[5, 6], [0, 1], [3, 4]]
        result = intersect_partitions(source, sink)
        assert result == [[0, 1], [3, 4], [5, 6]]

    def test_indices_sorted_within_groups(self):
        """Indices within each intersection group are sorted."""
        source = [[3, 1, 2], [6, 4, 5]]
        sink = [[3, 1, 2], [6, 4, 5]]
        result = intersect_partitions(source, sink)
        assert result == [[1, 2, 3], [4, 5, 6]]

    def test_non_contiguous_indices(self):
        """Works with non-contiguous index ranges when compatible."""
        source = [[0, 10, 20], [5, 15, 25]]
        # Sink has same grouping — compatible.
        sink = [[0, 10, 20], [5, 15, 25]]
        result = intersect_partitions(source, sink)
        assert result == [[0, 10, 20], [5, 15, 25]]

    def test_non_contiguous_indices_conflict(self):
        """Non-contiguous indices with conflicting constraints raises."""
        source = [[0, 10, 20], [5, 15, 25]]
        # Sink wants all together — incompatible.
        sink = [[0, 5, 10, 15, 20, 25]]
        with pytest.raises(ValueError):
            intersect_partitions(source, sink)

    def test_many_small_source_groups_incompatible_with_large_sink(self):
        """Many singleton source groups conflict with large sink groups."""
        source = [[i] for i in range(100)]  # 100 singleton groups
        sink = [list(range(0, 50)), list(range(50, 100))]  # 2 chunks
        # Sink requires 0-49 together but source separates them.
        with pytest.raises(ValueError, match="sink requires indices"):
            intersect_partitions(source, sink)

    def test_many_small_groups_compatible(self):
        """Many source groups with matching sink groups are compatible."""
        source = [[i] for i in range(100)]
        sink = [[i] for i in range(100)]
        result = intersect_partitions(source, sink)
        assert result is not None
        assert len(result) == 100
        assert all(len(g) == 1 for g in result)


# ---------------------------------------------------------------------------
# batch_groups
# ---------------------------------------------------------------------------


class TestBatchGroups:
    """Tests for batch_groups utility."""

    def test_fewer_groups_than_workers(self):
        """If groups <= workers, return groups unchanged."""
        groups = [[0, 1], [2, 3], [4, 5]]
        result = batch_groups(groups, n_workers=8)
        assert result == groups

    def test_equal_groups_and_workers(self):
        """If groups == workers, return groups unchanged."""
        groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
        result = batch_groups(groups, n_workers=4)
        assert result == groups

    def test_more_groups_than_workers(self):
        """Groups are merged into n_workers batches."""
        groups = [[0], [1], [2], [3], [4], [5], [6], [7]]
        result = batch_groups(groups, n_workers=2)
        assert len(result) == 2
        # All indices should be present.
        all_indices = sorted(idx for batch in result for idx in batch)
        assert all_indices == list(range(8))

    def test_load_balancing(self):
        """Greedy bin-packing balances groups by size."""
        # 1 big group + 3 small groups, 2 workers
        groups = [[0, 1, 2, 3, 4, 5], [6], [7], [8]]
        result = batch_groups(groups, n_workers=2)
        assert len(result) == 2
        # The big group goes in one batch, the 3 small in the other.
        sizes = sorted(len(b) for b in result)
        assert sizes == [3, 6]

    def test_preserves_all_indices(self):
        """All original indices appear exactly once in the result."""
        groups = [[0, 1], [2, 3, 4], [5], [6, 7], [8, 9]]
        result = batch_groups(groups, n_workers=3)
        all_indices = sorted(idx for batch in result for idx in batch)
        expected = sorted(idx for g in groups for idx in g)
        assert all_indices == expected

    def test_single_worker(self):
        """Single worker gets all indices in one batch."""
        groups = [[0, 1], [2, 3], [4, 5]]
        result = batch_groups(groups, n_workers=1)
        assert len(result) == 1
        all_indices = sorted(result[0])
        assert all_indices == [0, 1, 2, 3, 4, 5]

    def test_many_groups_few_workers(self):
        """50 groups distributed across 4 workers."""
        groups = [[i] for i in range(50)]
        result = batch_groups(groups, n_workers=4)
        assert len(result) == 4
        # Each batch should have ~12-13 indices.
        sizes = [len(b) for b in result]
        assert all(12 <= s <= 13 for s in sizes)
        # All indices present.
        all_indices = sorted(idx for batch in result for idx in batch)
        assert all_indices == list(range(50))

    def test_group_integrity_preserved(self):
        """Indices from same original group stay together in a batch."""
        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        result = batch_groups(groups, n_workers=2)
        # Each original group's indices must all be in the same batch.
        for group in groups:
            # Find which batch contains the first element.
            batch_idx = None
            for bi, batch in enumerate(result):
                if group[0] in batch:
                    batch_idx = bi
                    break
            assert batch_idx is not None
            # All elements of the group must be in the same batch.
            for idx in group:
                assert idx in result[batch_idx]
