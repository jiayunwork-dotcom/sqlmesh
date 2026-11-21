"""Validation and rejection tests for the metric filter feature."""
from __future__ import annotations

import pytest
from sqlglot import exp

from sqlmesh.core import dialect as d
from sqlmesh.core.metric.filter import Filter, _extract_regions
from sqlmesh.utils.errors import ConfigError, SQLMeshError


def _filter(sql: str) -> Filter:
    return Filter(d.parse_one(sql))


def test_accepts_supported_boolean_shape():
    f = _filter(
        "demo.tags.label = 'trial' "
        "AND (demo.activities.kind = 'promo' OR demo.activities.status = 'attended') "
        "OR NOT demo.sales.region IS NULL"
    )
    assert {m.split(".")[-1] for m in f.models} == {
        "tags",
        "activities",
        "sales",
    }


def test_rejects_function_call():
    with pytest.raises(ConfigError, match="Unsupported filter expression"):
        _filter("LOWER(demo.tags.label) = 'trial'")


def test_rejects_column_to_column_comparison():
    # A comparison whose two sides are both columns is not a constant predicate.
    with pytest.raises(ConfigError, match="single model|one column with one constant"):
        _filter("demo.sales.region = demo.tags.label")


def test_rejects_unqualified_column():
    with pytest.raises(ConfigError, match="fully qualified"):
        _filter("label = 'trial'")


def test_rejects_cross_model_leaf_predicate():
    with pytest.raises(ConfigError, match="single model"):
        _filter("demo.sales.customer_id = demo.tags.customer_id")


def test_rejects_non_boolean_constant():
    with pytest.raises(ConfigError, match="IS NULL"):
        # IS NULL against a constant rather than a column.
        _filter("'trial' IS NULL")


def test_region_same_model_predicate_kept_together():
    f = _filter(
        "demo.activities.kind = 'promo' "
        "AND demo.activities.status = 'attended'"
    )
    regions = _extract_regions(f.expression, f)
    assert len(regions) == 1
    assert regions[0].model.endswith("activities")


def test_region_splits_across_models():
    f = _filter(
        "demo.tags.label = 'trial' "
        "AND demo.activities.kind = 'promo' "
        "AND demo.activities.status = 'attended'"
    )
    regions = _extract_regions(f.expression, f)
    models = {r.model.split(".")[-1] for r in regions}
    assert models == {"tags", "activities"}
    # The two activities predicates remain one region.
    activity_regions = [r for r in regions if r.model.endswith("activities")]
    assert len(activity_regions) == 1


def test_not_is_part_of_region_predicate():
    f = _filter("NOT demo.tags.label IS NULL")
    regions = _extract_regions(f.expression, f)
    assert len(regions) == 1
    assert "NOT" in regions[0].expression.sql().upper()


def test_filter_reload_roundtrip():
    # Parsing a METRIC DDL keeps the filter attached and semantically equal.
    metric = d.parse_one(
        "METRIC (name m, expression SUM(demo.sales.amount), "
        "filter demo.tags.label = 'trial')"
    )
    filter_prop = next(
        prop for prop in metric.expressions if prop.name == "filter"
    )
    rebuilt = d.parse_one(metric.sql())
    rebuilt_prop = next(
        prop for prop in rebuilt.expressions if prop.name == "filter"
    )
    assert rebuilt_prop.args["value"].sql() == filter_prop.args["value"].sql()


def test_derived_metric_cannot_declare_filter():
    from sqlmesh.core.metric.definition import MetricMeta
    from sqlmesh.utils.errors import ConfigError as MetricConfigError

    base_a = MetricMeta(
        name="a", dialect="", expression=d.parse_one("SUM(demo.sales.amount)")
    )
    base_b = MetricMeta(
        name="b", dialect="", expression=d.parse_one("COUNT(demo.sales.id)")
    )
    derived = MetricMeta(
        name="bad",
        dialect="",
        expression=d.parse_one("a / b"),
        filter=d.parse_one("demo.tags.label = 'trial'"),
    )
    with pytest.raises(MetricConfigError, match="base metrics"):
        derived.to_metric({"a": base_a, "b": base_b}, {})


def test_filtered_metric_must_be_single_sum_count():
    from sqlmesh.core.metric.definition import MetricMeta

    # ARRAY_AGG is outside the supported filtered aggregation set.
    meta = MetricMeta(
        name="bad",
        dialect="",
        expression=d.parse_one("ARRAY_AGG(demo.sales.amount)"),
        filter=d.parse_one("demo.tags.label = 'trial'"),
    )
    with pytest.raises(ConfigError, match="unsupported aggregation|single aggregation"):
        meta.to_metric({}, {})


def test_filtered_metric_rejects_multi_model_measure():
    from sqlmesh.core.metric.definition import MetricMeta

    meta = MetricMeta(
        name="bad",
        dialect="",
        expression=d.parse_one(
            "SUM(IF(demo.customers.segment = 'x', demo.sales.amount, 0))"
        ),
        filter=d.parse_one("demo.tags.label = 'trial'"),
    )
    with pytest.raises(ConfigError, match="single fact model"):
        meta.to_metric({}, {})
