# Metric filter benchmark

This benchmark verifies two properties of the metric filter implementation at
scale:

1. A filter against a multi-valued related table never multiplies fact rows:
   every plan uses hash joins against eligibility subqueries grouped down to
   one row per fact key, and no `CROSS_PRODUCT` / nested-loop join appears.
2. Selective filters (few matching keys) and high-cardinality filters (many
   repeated rows per key) both stay fast because eligibility is pre-aggregated
   with `BOOL_OR` before touching the fact table.

## How to run

```bash
python benchmarks/metric_filters/benchmark.py --facts 100000 --related 1000000
```

Arguments:

* `--facts`: number of fact rows (default 100,000).
* `--related`: number of rows in the high-cardinality related table (default
  1,000,000, roughly 500 rows per customer).

The script generates a throwaway SQLMesh project under a temporary directory,
applies it to an embedded DuckDB database, and prints for each query:

* the result rows,
* execution time,
* the count of cartesian-product operators in the physical plan (must be 0),
* an excerpt of the plan showing the LEFT hash joins to grouped eligibility
  subqueries.

## Observed behavior on the default workload (100k facts / 1M related rows)

All queries complete in roughly 0.1 seconds or less, and every plan shows:

* `HASH_JOIN ... Join Type: LEFT` between the fact table and one eligibility
  subquery per related model,
* the eligibility subqueries grouped by the fact join key with `BOOL_OR`
  predicates,
* zero cartesian-product operators.

Because each eligibility subquery returns at most one row per fact key, the
fact table's row count is unchanged by any number of matching related rows.
