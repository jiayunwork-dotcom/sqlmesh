"""Scale and query-plan benchmark for metric filters on DuckDB.

Generates:
  * 100,000 fact rows
  * ~1,000,000 high-cardinality related rows (many repeated rows per fact key)
  * a smaller related table with selective, low-cardinality predicates

The script prints timings, peak DuckDB memory and the generated query plan so
the absence of cartesian products can be verified from the operators. It is
self-contained and writes everything into a temporary DuckDB database.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import time
from pathlib import Path

FIXTURE_ROOT = Path(__file__).parent / "fixture_project"


def write_seeds(root: Path, fact_rows: int, related_rows: int) -> None:
    seeds = root / "seeds"
    seeds.mkdir(parents=True, exist_ok=True)

    with open(seeds / "big_facts.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fact_id", "customer_id", "region", "amount"])
        for fact_id in range(1, fact_rows + 1):
            customer_id = (fact_id % 2000) + 1
            region = ("EAST", "WEST", "NORTH", "SOUTH")[fact_id % 4]
            writer.writerow([fact_id, customer_id, region, fact_id % 1000])

    # High-cardinality multi-valued table: ~500 rows per customer on average.
    with open(seeds / "big_events.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["customer_id", "kind", "status"])
        customers = 2000
        per_customer = max(1, related_rows // customers)
        for customer_id in range(1, customers + 1):
            for i in range(per_customer):
                kind = "webinar" if i % 2 else "promo"
                status = "attended" if i % 3 else "invited"
                writer.writerow([customer_id, kind, status])

    # Selective low-cardinality table with few matching customers.
    with open(seeds / "big_tags.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["customer_id", "label"])
        for customer_id in range(1, customers + 1):
            # Every customer gets several tags but only 10% carry 'trial'.
            writer.writerow([customer_id, "paid"])
            writer.writerow([customer_id, "paid"])
            if customer_id % 10 == 0:
                writer.writerow([customer_id, "trial"])
                writer.writerow([customer_id, "trial"])


def write_project(root: Path) -> None:
    (root / "models").mkdir(exist_ok=True)
    (root / "metrics").mkdir(exist_ok=True)

    (root / "config.yaml").write_text(
        """
gateways:
  duckdb:
    connection:
      type: duckdb
      database: benchmark.duckdb
default_gateway: duckdb
model_defaults:
  dialect: duckdb
"""
    )

    (root / "models" / "big_facts.sql").write_text(
        """
MODEL (
  name bench.big_facts,
  kind SEED (path '../seeds/big_facts.csv'),
  grain fact_id,
  references customer_id
);
"""
    )
    (root / "models" / "big_events.sql").write_text(
        """
MODEL (
  name bench.big_events,
  kind SEED (path '../seeds/big_events.csv'),
  grain (customer_id, kind, status),
  references customer_id
);
"""
    )
    (root / "models" / "big_tags.sql").write_text(
        """
MODEL (
  name bench.big_tags,
  kind SEED (path '../seeds/big_tags.csv'),
  grain (customer_id, label),
  references customer_id
);
"""
    )

    (root / "metrics" / "metrics.sql").write_text(
        """
METRIC (
  name total_amount,
  expression SUM(benchmark.bench.big_facts.amount)
);

METRIC (
  name trial_amount,
  expression SUM(benchmark.bench.big_facts.amount),
  filter benchmark.bench.big_tags.label = 'trial'
);

METRIC (
  name promo_attended_amount,
  expression SUM(benchmark.bench.big_facts.amount),
  filter benchmark.bench.big_events.kind = 'promo'
    AND benchmark.bench.big_events.status = 'attended'
);

METRIC (
  name trial_promo_amount,
  expression SUM(benchmark.bench.big_facts.amount),
  filter benchmark.bench.big_tags.label = 'trial'
    AND benchmark.bench.big_events.kind = 'promo'
    AND benchmark.bench.big_events.status = 'attended'
);

METRIC (
  name promo_order_count,
  expression COUNT(benchmark.bench.big_facts.fact_id),
  filter benchmark.bench.big_events.kind = 'promo'
    AND benchmark.bench.big_events.status = 'attended'
);
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=int, default=100_000)
    parser.add_argument("--related", type=int, default=1_000_000)
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="metric_bench_"))
    try:
        root = workdir / "project"
        shutil.copytree(FIXTURE_ROOT, root, dirs_exist_ok=True)
        write_seeds(root, args.facts, args.related)
        write_project(root)

        from sqlmesh.core.context import Context

        started = time.perf_counter()
        ctx = Context(paths=str(root))
        ctx.plan("prod", auto_apply=True, no_prompts=True, skip_tests=True)
        load_seconds = time.perf_counter() - started
        print(f"project load + apply: {load_seconds:.2f}s")

        queries = {
            "total": "SELECT METRIC(total_amount) FROM benchmark.bench.big_facts",
            "trial_filter": "SELECT METRIC(trial_amount) FROM benchmark.bench.big_facts",
            "promo_filter": "SELECT METRIC(promo_attended_amount) FROM benchmark.bench.big_facts",
            "two_multivalued": "SELECT METRIC(trial_promo_amount) FROM benchmark.bench.big_facts",
            "grouped_trial": (
                "SELECT region, METRIC(total_amount), METRIC(trial_amount), "
                "METRIC(promo_attended_amount) "
                "FROM benchmark.bench.big_facts GROUP BY region ORDER BY region"
            ),
        }

        # Peak memory tracking for the connection.
        ctx.engine_adapter.cursor.execute("SET memory_limit='2GB'")

        for label, sql in queries.items():
            rewritten = ctx.rewrite(sql).sql(dialect="duckdb")
            plan = ctx.engine_adapter.cursor.execute(
                "EXPLAIN " + rewritten
            ).fetchall()
            plan_text = plan[0][1] if plan else ""

            started = time.perf_counter()
            rows = ctx.engine_adapter.cursor.execute(rewritten).fetchall()
            elapsed = time.perf_counter() - started

            # Count the actual intermediate row estimates and check for
            # cartesian-product signatures.
            product_signals = plan_text.count("CROSS_PRODUCT") + plan_text.count(
                "NESTED_LOOP_JOIN"
            )
            print(f"\n=== {label} ===")
            print(f"rows: {rows}")
            print(f"elapsed: {elapsed:.3f}s")
            print(f"cartesian product operators in plan: {product_signals}")
            print("plan excerpt:")
            print("\n".join(plan_text.splitlines()[:25]))

        db_path = root / "benchmark.duckdb"
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            print(f"\nduckdb database size: {size_mb:.1f} MB")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
