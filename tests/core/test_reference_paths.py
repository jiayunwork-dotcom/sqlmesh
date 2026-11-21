"""Reference graph rejection tests for eligibility path resolution."""
from __future__ import annotations

import pytest
from sqlglot import exp as sqlglot_exp

from sqlmesh.core.reference import Reference, ReferenceGraph
from sqlmesh.utils.errors import SQLMeshError


class _RawExpression:
    """Wrapper so a reference can be supplied as an already-parsed expression."""


class _Model:
    def __init__(self, name: str, grains, references):
        self.name = name
        self.grains = [_exp(ref) for ref in grains]
        ref_objects = []
        for ref in references:
            if isinstance(ref, tuple):
                ref_name, unique = ref
                ref_objects.append(_mock_ref(name, ref_name, unique))
            elif isinstance(ref, _RawExpression):
                ref_objects.append(
                    Reference(model_name=name, expression=ref.expression)
                )
            else:
                ref_objects.append(
                    Reference(model_name=name, expression=_exp(ref))
                )
        self.all_references = ref_objects
        self.columns_to_types = {}


def _exp(sql: str):
    from sqlglot import parse_one

    return parse_one(sql)


def raw(expression: str) -> _RawExpression:
    marker = _RawExpression()
    marker.expression = _exp(expression)
    return marker


def _mock_ref(model_name: str, ref_name: str, unique: bool):
    from unittest.mock import Mock

    ref = Mock()
    ref.name = ref_name
    ref.model_name = model_name
    ref.unique = unique
    return ref


def _graph(models):
    return ReferenceGraph(models)


def test_unknown_model_raises_locatable_error():
    graph = _graph([_Model("a", ["id"], ["id"])])
    with pytest.raises(SQLMeshError, match="Unknown model"):
        graph.find_eligibility_path("a", "missing")


def test_no_reference_path_raises():
    # Two disconnected model groups sharing no reference name.
    graph = _graph(
        [
            _Model("fact", ["fid"], [("k", True)]),
            _Model("island", ["iid"], [("other_key", True)]),
        ]
    )
    with pytest.raises(SQLMeshError, match="Cannot find path"):
        graph.find_eligibility_path("fact", "island")


def test_ambiguous_shortest_paths_rejected():
    # Both target models expose the same reference name from the source, so the
    # graph cannot choose a unique path to a specific named target when two
    # equally short candidates reach it. Build two routes into one target.
    graph = _graph(
        [
            _Model("fact", ["fid"], ["shared_key", "other_key"]),
            _Model("middle", ["mid"], ["shared_key", "join_key"]),
            _Model("target", ["tid"], ["other_key", "join_key"]),
        ]
    )
    # Two equal-length routes into the target: fact->m1->target and
    # fact->m2->target. Both intermediate models are unique on their incoming
    # key, so both paths are valid and the graph must refuse to choose.
    graph = _graph(
        [
            _Model("fact", ["fid"], ["k1", "k2"]),
            _Model("m1", ["k1"], ["k1", "join_key"]),
            _Model("m2", ["k2"], ["k2", "join_key"]),
            _Model("target", ["tid"], ["join_key"]),
        ]
    )
    with pytest.raises(SQLMeshError, match="Cannot uniquely resolve"):
        graph.find_eligibility_path("fact", "target")


def test_regular_join_rejects_many_to_many_terminal_hop():
    # The target is not on the unique side of the shared reference, so a
    # regular measure join that fans out must be rejected.
    graph = _graph(
        [
            _Model("fact", ["fid"], [("customer_id", True)]),
            _Model("events", ["eid"], [("customer_id", False)]),
        ]
    )
    # Give the target a non-unique grain relative to the key by leaving
    # customer_id out of its grain.
    target = next(m for m in [_Model("events", ["eid"], ["customer_id"])])
    graph = _graph(
        [
            _Model("fact", ["fid"], ["customer_id"]),
            target,
        ]
    )
    with pytest.raises(SQLMeshError, match="Cannot find path"):
        graph.find_path("fact", "events")


def test_eligibility_allows_terminal_many_to_many():
    graph = _graph(
        [
            _Model("fact", ["fid"], ["customer_id"]),
            _Model("events", ["eid", "kind"], ["customer_id"]),
        ]
    )
    path = graph.find_eligibility_path("fact", "events")
    assert [ref.model_name for ref in path] == ["fact", "events"]
    # The destination reference resolves to the shared key columns.
    assert path[-1].name == path[0].name
    assert "customer_id" in path[-1].name


def test_eligibility_rejects_intermediate_many_to_many():
    # fact -> bridge (many side) -> target. The many-side hop is not terminal,
    # so eligibility cannot use it either. The bridge's grain includes its
    # own id but not the fact key, making the hop non-unique.
    graph = _graph(
        [
            _Model("fact", ["fid"], ["customer_id"]),
            _Model("bridge", ["bid"], ["customer_id", "target_id"]),
            _Model("target", ["tid"], ["target_id"]),
        ]
    )
    with pytest.raises(SQLMeshError, match="Cannot find path"):
        graph.find_eligibility_path("fact", "target")


def test_composite_reference_key_is_preserved():
    fact = _Model("fact", ["fid"], [raw("(a, b)")])
    target = _Model("target", ["tid"], [raw("(a, b)")])
    graph = _graph([fact, target])
    path = graph.find_eligibility_path("fact", "target")
    assert path[-1].columns == ["a", "b"]
