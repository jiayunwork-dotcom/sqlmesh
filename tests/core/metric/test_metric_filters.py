"""End-to-end verification of metric filters on DuckDB.

Expected values are computed independently by iterating the raw fact rows and
the seed CSVs in pure Python; the SQLMesh rewriter is never used to derive an
answer, only to produce the query under test.
"""
from __future__ import annotations

import csv
import shutil
import typing as t
from pathlib import Path

import pytest
from sqlmesh.core.context import Context

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "metric_filters"
SEEDS = FIXTURE_ROOT / "seeds"


def _load_csv(name: str) -> t.List[t.Dict[str, str]]:
    with open(SEEDS / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _customer(value: str) -> t.Optional[int]:
    return int(value) if value else None


def _region(value: str) -> str:
    return value or "NULL"


@pytest.fixture(scope="module")
def context(tmp_path_factory) -> Context:
    project = tmp_path_factory.mktemp("metric_filters")
    shutil.copytree(FIXTURE_ROOT, project, dirs_exist_ok=True)
    ctx = Context(paths=str(project))
    ctx.plan("prod", auto_apply=True, no_prompts=True, skip_tests=True)
    return ctx


@pytest.fixture(scope="module")
def expected() -> t.Dict[str, t.Any]:
    sales = _load_csv("sales.csv")
    refunds = _load_csv("refunds.csv")
    tags = _load_csv("tags.csv")
    activities = _load_csv("activities.csv")

    trial_customers = {
        _customer(r["customer_id"])
        for r in tags
        if r["label"] == "trial" and r["customer_id"]
    }
    promo_attendees = {
        _customer(r["customer_id"])
        for r in activities
        if r["kind"] == "promo" and r["status"] == "attended" and r["customer_id"]
    }
    # Same-row "webinar AND invited": only customer 4 has such a record.
    webinar_invited = {
        _customer(r["customer_id"])
        for r in activities
        if r["kind"] == "webinar" and r["status"] == "invited" and r["customer_id"]
    }
    # The stitched (incorrect) set: customer 2 would appear if halves from
    # different rows were combined. It must NOT be in webinar_invited.
    promo_kinds = {
        _customer(r["customer_id"])
        for r in activities
        if r["kind"] == "promo" and r["customer_id"]
    }
    invited_statuses = {
        _customer(r["customer_id"])
        for r in activities
        if r["status"] == "invited" and r["customer_id"]
    }
    stitched = promo_kinds & invited_statuses

    metrics = (
        "total_sales",
        "trial_sales",
        "promo_attendee_sales",
        "trial_promo_sales",
        "west_or_promo_sales",
        "half_condition_sales",
    )

    def eligible(row: t.Dict[str, str], metric: str) -> bool:
        customer = _customer(row["customer_id"])
        if metric == "total_sales":
            return True
        if metric == "trial_sales":
            return customer in trial_customers
        if metric == "promo_attendee_sales":
            return customer in promo_attendees
        if metric == "trial_promo_sales":
            return customer in trial_customers and customer in promo_attendees
        if metric == "west_or_promo_sales":
            return row["region"] == "WEST" or customer in promo_attendees
        if metric == "half_condition_sales":
            return customer in webinar_invited
        raise AssertionError(metric)

    sums: t.Dict[str, t.Dict[str, int]] = {}
    counts: t.Dict[str, t.Dict[str, int]] = {}

    for row in sales:
        region = _region(row["region"])
        sum_bucket = sums.setdefault(region, {})
        count_bucket = counts.setdefault(region, {})

        count_bucket["order_count"] = count_bucket.get("order_count", 0) + 1
        for metric in metrics:
            if eligible(row, metric):
                sum_bucket[metric] = sum_bucket.get(metric, 0) + int(row["amount"])
        if _customer(row["customer_id"]) in trial_customers:
            count_bucket["trial_order_count"] = (
                count_bucket.get("trial_order_count", 0) + 1
            )
        if _customer(row["customer_id"]) in promo_attendees:
            count_bucket["promo_order_count"] = (
                count_bucket.get("promo_order_count", 0) + 1
            )

    for bucket in counts.values():
        bucket.setdefault("trial_order_count", 0)
        bucket.setdefault("promo_order_count", 0)

    grand_total = sum(int(r["amount"]) for r in sales)
    trial_total = sum(
        int(r["amount"]) for r in sales if _customer(r["customer_id"]) in trial_customers
    )
    distinct_customers = {
        _customer(r["customer_id"]) for r in sales if r["customer_id"]
    }
    refund_total = sum(int(r["amount"]) for r in refunds)
    # Refunds of trial-tagged customers: customer 2 (refund 10) only. Customer
    # 4's refund (40) is not trial because customer 4... is trial-tagged, so it
    # is included — compute from the tags rather than hard-coding.
    trial_refund = sum(
        int(r["amount"])
        for r in refunds
        if _customer(r["customer_id"]) in trial_customers
    )

    return {
        "sums_by_region": sums,
        "counts_by_region": counts,
        "grand_total_sales": grand_total,
        "trial_total_sales": trial_total,
        "distinct_customers": distinct_customers,
        "refund_total": refund_total,
        "trial_refund": trial_refund,
        "trial_customers": trial_customers,
        "promo_attendees": promo_attendees,
        "webinar_invited": webinar_invited,
        "stitched_wrongly": stitched,
    }


def _execute(context: Context, sql: str) -> t.Tuple[t.List[str], t.List[t.Tuple]]:
    rewritten = context.rewrite(sql).sql(dialect="duckdb")
    cursor = context.engine_adapter.cursor.execute(rewritten)
    return [desc[0] for desc in cursor.description], cursor.fetchall()


def _one_row(context: Context, sql: str) -> t.Dict[str, t.Any]:
    columns, rows = _execute(context, sql)
    assert len(rows) == 1, rows
    return dict(zip(columns, rows[0]))


def _grouped(context: Context, sql: str) -> t.List[t.Tuple]:
    """Return grouped rows preserving duplicates of NULL group keys."""
    _, rows = _execute(context, sql)
    return rows


def _group_map(context: Context, sql: str) -> t.Dict[str, t.Tuple]:
    return {
        (row[0] if row[0] is not None else "NULL"): row[1:]
        for row in _grouped(context, sql)
    }


def test_repeated_tags_do_not_multiply_amounts(context, expected):
    # Customer 1 carries two identical 'trial' tag rows; its sales count once.
    rows = _one_row(
        context,
        "SELECT METRIC(trial_sales), METRIC(total_sales) FROM metric_filters.demo.sales",
    )
    assert rows["trial_sales"] == expected["trial_total_sales"]
    assert rows["total_sales"] == expected["grand_total_sales"]


def test_grouped_sales_metrics_match_row_by_row_judgement(context, expected):
    result = _group_map(
        context,
        """
        SELECT region,
               METRIC(total_sales),
               METRIC(trial_sales),
               METRIC(promo_attendee_sales),
               METRIC(trial_promo_sales),
               METRIC(west_or_promo_sales),
               METRIC(half_condition_sales)
        FROM metric_filters.demo.sales
        GROUP BY region
        ORDER BY region
        """,
    )
    metric_names = (
        "total_sales",
        "trial_sales",
        "promo_attendee_sales",
        "trial_promo_sales",
        "west_or_promo_sales",
        "half_condition_sales",
    )
    for region, values in result.items():
        want_sums = expected["sums_by_region"][region]
        for i, metric in enumerate(metric_names):
            assert values[i] == want_sums.get(metric), (region, metric, values[i])


def test_count_zero_and_sum_null_on_empty_group(context, expected):
    rows = _group_map(
        context,
        """
        SELECT region,
               METRIC(order_count),
               METRIC(trial_order_count),
               METRIC(promo_order_count),
               METRIC(trial_sales)
        FROM metric_filters.demo.sales
        GROUP BY region
        ORDER BY region
        """,
    )
    west = rows["WEST"]
    assert west[0] == expected["counts_by_region"]["WEST"]["order_count"]
    # Customer 2 carries the trial tag, so WEST has one trial order but no
    # promo-attendee orders.
    assert west[1] == 1
    assert west[2] == 0
    assert west[3] == 50  # customer 2's WEST sale is trial-eligible


def test_two_multivalued_tables_no_cartesian_product(context, expected):
    # Customer 1 has 3 tags and 2 activities. A cartesian product would
    # multiply its amounts; the filtered total equals the Python judgement.
    rows = _one_row(
        context,
        "SELECT METRIC(trial_promo_sales) FROM metric_filters.demo.sales",
    )
    want = sum(
        int(r["amount"])
        for r in _load_csv("sales.csv")
        if _customer(r["customer_id"]) in expected["trial_customers"]
        and _customer(r["customer_id"]) in expected["promo_attendees"]
    )
    assert rows["trial_promo_sales"] == want


def test_half_conditions_cannot_be_stitched_across_rows(context):
    # Customer 2 would qualify only by combining two different activity rows;
    # same-row semantics must exclude it. Customer 4 has one matching row.
    rows = _one_row(
        context,
        "SELECT METRIC(half_condition_sales) FROM metric_filters.demo.sales",
    )
    assert rows["half_condition_sales"] == 300


def test_or_with_missing_related_rows_keeps_fact_side(context):
    # The OR combines the fact-side WEST predicate with the promo-attendee
    # population. WEST sales (125) qualify directly even though customers 2 and
    # 3 have no attended-promo row; promo attendees (customer 1, sales 100+200;
    # customer 4, sale 300) qualify via the related table.
    rows = _one_row(
        context,
        "SELECT METRIC(west_or_promo_sales) FROM metric_filters.demo.sales",
    )
    assert rows["west_or_promo_sales"] == 725


def test_missing_related_object_is_all_null_row(context):
    # Customer 3 has no activity rows at all. Under three valued logic the
    # activity region evaluates to NULL for their sales, so an AND-only filter
    # must exclude them while the OR filter above still admits them through the
    # fact-side branch.
    rows = _one_row(
        context,
        "SELECT METRIC(promo_attendee_sales) FROM metric_filters.demo.sales",
    )
    # Customer 1 (300) + customer 4 (300); customer 3's WEST 75 is excluded.
    assert rows["promo_attendee_sales"] == 600


def test_null_join_keys_do_not_match(context, expected):
    # The NULL-customer sale (sale 8) has a NULL key; SQL equality never matches
    # it to the NULL-customer tag/activity rows. It stays in total_sales only.
    rows = _one_row(
        context,
        "SELECT METRIC(total_sales), METRIC(trial_sales) FROM metric_filters.demo.sales",
    )
    assert rows["total_sales"] == expected["grand_total_sales"]
    assert rows["trial_sales"] == expected["trial_total_sales"]
    # The NULL-key sale (amount 500) must not be included in the trial total.
    assert rows["total_sales"] - rows["trial_sales"] >= 500


def test_null_grouping_values_merge_across_facts(context):
    # Both facts have a NULL-region row. Their NULL group keys must merge into
    # one output row (one row carrying both facts' measures), while the
    # refund-only SOUTH region survives as a separate row with NULL sales.
    rows = _grouped(
        context,
        """
        SELECT region,
               METRIC(total_sales),
               METRIC(trial_sales),
               METRIC(total_refunds),
               METRIC(trial_refunds)
        FROM metric_filters.demo.sales
        GROUP BY region
        ORDER BY region
        """,
    )
    # Column layout: (region, total_sales, trial_sales, total_refunds,
    # trial_refunds).
    refund_only_null = next(
        row for row in rows if row[0] is None and row[1] is None
    )
    # Refund-only SOUTH region: its amount is 15 and customer 8 is not
    # trial-tagged, so the filtered refund is NULL.
    assert refund_only_null[3] == 15
    assert refund_only_null[4] is None

    combined_null = next(
        row for row in rows if row[0] is None and row[1] is not None
    )
    assert combined_null[1] == 400
    assert combined_null[2] == 400  # customer 6 is trial-tagged
    assert combined_null[3] == 20  # the NULL-region refund
    assert combined_null[4] == 20  # customer 6 is trial-tagged too
    # EAST, WEST, the merged NULL group, and refund-only SOUTH.
    assert len(rows) == 4


def test_groups_owned_by_only_one_fact(context):
    # Refund region SOUTH exists only on the refunds fact and appears when that
    # fact drives the FROM clause; it never appears on the sales side.
    refund_rows = _group_map(
        context,
        """
        SELECT region,
               METRIC(total_refunds),
               METRIC(trial_refunds)
        FROM metric_filters.demo.refunds
        GROUP BY region
        ORDER BY region
        """,
    )
    assert "SOUTH" in refund_rows
    assert refund_rows["SOUTH"][0] == 15
    assert refund_rows["SOUTH"][1] is None  # customer 8 is not trial-tagged

    sales_rows = _group_map(
        context,
        """
        SELECT region, METRIC(total_sales)
        FROM metric_filters.demo.sales
        GROUP BY region
        """,
    )
    assert "SOUTH" not in sales_rows


def test_refund_fact_filter(context, expected):
    rows = _one_row(
        context,
        "SELECT METRIC(total_refunds), METRIC(trial_refunds) FROM metric_filters.demo.refunds",
    )
    assert rows["total_refunds"] == expected["refund_total"]
    assert rows["trial_refunds"] == expected["trial_refund"]


def test_derived_ratios(context, expected):
    rows = _one_row(
        context,
        "SELECT METRIC(trial_sales_share), METRIC(refund_rate) FROM metric_filters.demo.sales",
    )
    assert rows["trial_sales_share"] == pytest.approx(
        expected["trial_total_sales"] / expected["grand_total_sales"]
    )
    assert rows["refund_rate"] == pytest.approx(
        expected["refund_total"] / expected["grand_total_sales"]
    )


def test_derived_ratio_zero_divisor_is_null(context):
    rows = _group_map(
        context,
        """
        SELECT region, METRIC(promo_to_trial_ratio)
        FROM metric_filters.demo.sales
        GROUP BY region
        ORDER BY region
        """,
    )
    # EAST: 3 promo orders / 3 trial orders = 1.
    assert rows["EAST"][0] == 1.0
    # WEST: 0 promo orders / 1 trial order = 0 (no error).
    assert rows["WEST"][0] == 0.0
    # NULL region: 0 promo / 1 trial = 0.
    assert rows["NULL"][0] == 0.0

    # In WEST both populations behind empty_to_empty_ratio are empty: the
    # divisor SUM is NULL and the division must return NULL, not raise.
    empty_rows = _group_map(
        context,
        """
        SELECT region, METRIC(empty_to_empty_ratio)
        FROM metric_filters.demo.sales
        GROUP BY region
        ORDER BY region
        """,
    )
    assert empty_rows["WEST"][0] is None

    # The generated SQL guards division with NULLIF.
    rewritten = context.rewrite(
        "SELECT METRIC(promo_to_trial_ratio) FROM metric_filters.demo.sales"
    ).sql(dialect="duckdb")
    assert "NULLIF" in rewritten.upper()


def test_distinct_count_filtered(context, expected):
    rows = _one_row(
        context,
        "SELECT METRIC(distinct_sale_customers) FROM metric_filters.demo.sales",
    )
    # COUNT DISTINCT ignores the NULL-customer row.
    assert rows["distinct_sale_customers"] == len(expected["distinct_customers"])


def test_where_clause_applies_to_every_metric(context):
    rows = _one_row(
        context,
        """
        SELECT METRIC(total_sales), METRIC(trial_sales)
        FROM metric_filters.demo.sales
        WHERE region = 'EAST'
        """,
    )
    # EAST total = 100+200+300+25+500 = 1125; trial EAST = 100+200+300 = 600.
    assert rows["total_sales"] == 1125
    assert rows["trial_sales"] == 600
