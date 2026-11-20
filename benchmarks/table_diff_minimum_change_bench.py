#!/usr/bin/env python3
"""Scalability and boundary experiments for the ``minimum_change`` table diff.

This script is *not* part of the unit test suite: it deliberately builds large temporary
DuckDB tables so the host/database split of the minimum-change comparison can be measured.
Run it with the project's virtual environment, for example::

    .venv/bin/python -I benchmarks/table_diff_minimum_change_bench.py

Every table is created inside a temporary DuckDB database file and dropped at the end. The
script reports:

* the number of SQL statements issued and the number of rows transferred to the host
  (fetchdf / fetchall / fetchone calls);
* the host-side peak Python allocation (``tracemalloc``) and wall-clock time;
* the exact diff statistics, including the total changed-cell count;
* the over-cap rejection, single-side large multiplicity and mid-query failure boundaries.

The host payload is expected to stay bounded by the per-key residual cap
(``max_unmatched_rows``), the configured sample limit and fixed-size aggregation buffers,
regardless of the number of business keys or the total row count.
"""

from __future__ import annotations

import tracemalloc
import typing as t
from pathlib import Path
from tempfile import TemporaryDirectory


from sqlmesh.core.config import DuckDBConnectionConfig
from sqlmesh.core.engine_adapter.base import EngineAdapter
from sqlmesh.core.table_diff import TableDiff
from sqlmesh.utils.errors import SQLMeshError

HOT_KEY_COPIES = 100_000
DISTINCT_KEYS = 99_990
REAL_CHANGES = 10
TOTAL_ROWS = 2 * HOT_KEY_COPIES + DISTINCT_KEYS * 2


class QueryCounter:
    """Intercepts an adapter's data-returning calls to measure host transfer volume."""

    def __init__(self, adapter: EngineAdapter) -> None:
        self.adapter = adapter
        self.executed = 0
        self.fetch_calls = 0
        self.rows_transferred = 0
        self._orig_execute = adapter._execute
        self._orig_fetchdf = adapter.fetchdf
        self._orig_fetchall = adapter.fetchall
        self._orig_fetchone = adapter.fetchone

    def __enter__(self) -> "QueryCounter":
        adapter = self.adapter

        def _execute(sql: str, *args: t.Any, **kwargs: t.Any) -> None:
            self.executed += 1
            return self._orig_execute(sql, *args, **kwargs)

        def fetchdf(query: t.Any, *args: t.Any, **kwargs: t.Any) -> t.Any:
            df = self._orig_fetchdf(query, *args, **kwargs)
            self.fetch_calls += 1
            self.rows_transferred += len(df)
            return df

        def fetchall(query: t.Any, *args: t.Any, **kwargs: t.Any) -> t.List[t.Tuple]:
            rows = self._orig_fetchall(query, *args, **kwargs)
            self.fetch_calls += 1
            self.rows_transferred += len(rows)
            return rows

        def fetchone(query: t.Any, *args: t.Any, **kwargs: t.Any) -> t.Any:
            self.fetch_calls += 1
            return self._orig_fetchone(query, *args, **kwargs)

        adapter._execute = _execute  # type: ignore[method-assign]
        adapter.fetchdf = fetchdf  # type: ignore[method-assign]
        adapter.fetchall = fetchall  # type: ignore[method-assign]
        adapter.fetchone = fetchone  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: t.Any) -> None:
        adapter = self.adapter
        adapter._execute = self._orig_execute  # type: ignore[method-assign]
        adapter.fetchdf = self._orig_fetchdf  # type: ignore[method-assign]
        adapter.fetchall = self._orig_fetchall  # type: ignore[method-assign]
        adapter.fetchone = self._orig_fetchone  # type: ignore[method-assign]


def build_large_tables(adapter: EngineAdapter) -> None:
    """~200k+ rows: one hot key with 100k identical copies, many distinct keys, few changes."""
    adapter.execute(
        """
        CREATE TABLE bench_src (id BIGINT, grp VARCHAR, v INTEGER, d DOUBLE);
        CREATE TABLE bench_tgt (id BIGINT, grp VARCHAR, v INTEGER, d DOUBLE);
        """
    )
    # The hot key is identical on both sides: 100k fully-equal copies each.
    adapter.execute(
        f"""
        INSERT INTO bench_src
        SELECT 0, 'hot', 7, 1.5 FROM range({HOT_KEY_COPIES});
        INSERT INTO bench_tgt
        SELECT 0, 'hot', 7, 1.5 FROM range({HOT_KEY_COPIES});
        """
    )
    # Many distinct keys, identical on both sides.
    adapter.execute(
        f"""
        INSERT INTO bench_src
        SELECT i, 'distinct_' || i, i, i + 0.25 FROM range({DISTINCT_KEYS}) t(i);
        INSERT INTO bench_tgt
        SELECT i, 'distinct_' || i, i, i + 0.25 FROM range({DISTINCT_KEYS}) t(i);
        """
    )
    # A handful of genuine changes, each a two-sided residual within its key.
    for i in range(REAL_CHANGES):
        key = 1_000_000 + i
        adapter.execute(
            f"INSERT INTO bench_src VALUES ({key}, 'change', {i}, 2.0), ({key}, 'change', {i + 1}, 2.0)"
        )
        adapter.execute(
            f"INSERT INTO bench_tgt VALUES ({key}, 'change', {i + 50}, 2.0), ({key}, 'change', {i + 1}, 2.0)"
        )


def report(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    with TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "bench.duckdb")
        adapter = DuckDBConnectionConfig(database=db_path).create_engine_adapter()

        build_large_tables(adapter)
        src_count = adapter.fetchone("SELECT COUNT(*) FROM bench_src")[0]
        tgt_count = adapter.fetchone("SELECT COUNT(*) FROM bench_tgt")[0]
        print(f"source rows: {src_count:,}")
        print(f"target rows: {tgt_count:,}")

        # 1) Full minimum-change diff on the large tables.
        report("large-table minimum-change diff")
        td = TableDiff(
            adapter,
            "bench_src",
            "bench_tgt",
            on=["id", "grp"],
            row_match="minimum_change",
            limit=20,
        )
        tracemalloc.start()
        counter = QueryCounter(adapter)
        with counter:
            import time

            start = time.perf_counter()
            rd = td.row_diff()
            elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"elapsed seconds:                 {elapsed:.3f}")
        print(f"SQL statements executed:         {counter.executed}")
        print(f"data fetch calls to host:        {counter.fetch_calls}")
        print(f"rows transferred to host:       {counter.rows_transferred:,}")
        print(f"host peak Python allocation:     {peak / 1024 / 1024:.2f} MiB")
        print(f"source count / target count:     {rd.source_count:,} / {rd.target_count:,}")
        print(f"full matches:                    {rd.full_match_count:,}")
        print(f"changed pairings:                {rd.partial_match_count:,}")
        print(f"source-only / target-only:       {rd.s_only_count:,} / {rd.t_only_count:,}")
        print(f"total changed cells:             {rd.changed_cell_count:,}")
        assert rd.source_count == rd.full_match_count + rd.partial_match_count + rd.s_only_count
        assert rd.target_count == rd.full_match_count + rd.partial_match_count + rd.t_only_count
        # Only the 10 changed keys transfer residual rows; the 100k hot key never leaves the DB.
        assert counter.rows_transferred < 1_000, (
            "host transfer must be bounded by the residual cap and sample limit, "
            f"got {counter.rows_transferred} rows"
        )
        leftovers = adapter.fetchall(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'sqlmesh_temp'"
        )
        assert leftovers == [], f"temp tables leaked: {leftovers}"

        # 2) Two-sided key that exceeds the residual cap must abort with a clear message.
        report("two-sided key exceeding max_unmatched_rows")
        adapter.execute(
            "CREATE TABLE cap_src (id INT, v INT); CREATE TABLE cap_tgt (id INT, v INT)"
        )
        adapter.execute(
            "INSERT INTO cap_src SELECT 1, i FROM range(130) t(i); "
            "INSERT INTO cap_tgt SELECT 1, i + 500 FROM range(130) t(i)"
        )
        cap_td = TableDiff(
            adapter,
            "cap_src",
            "cap_tgt",
            on=["id"],
            row_match="minimum_change",
            max_unmatched_rows=128,
        )
        try:
            cap_td.row_diff()
        except SQLMeshError as e:
            print(f"rejected as expected: {str(e)[:160]}...")
        else:
            raise AssertionError("expected an over-cap SQLMeshError")
        leftovers = adapter.fetchall(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'sqlmesh_temp'"
        )
        assert leftovers == [], f"temp tables leaked after cap abort: {leftovers}"

        # 3) A single-side group with a huge multiplicity is reported by count, never expanded.
        report("single-side large multiplicity")
        adapter.execute(
            "CREATE TABLE one_src (id INT, v INT); CREATE TABLE one_tgt (id INT, v INT)"
        )
        adapter.execute("INSERT INTO one_src SELECT 1, i FROM range(500) t(i)")
        one_td = TableDiff(
            adapter,
            "one_src",
            "one_tgt",
            on=["id"],
            row_match="minimum_change",
            max_unmatched_rows=8,
        )
        one_counter = QueryCounter(adapter)
        with one_counter:
            one_rd = one_td.row_diff()
        print(f"source-only copies:              {one_rd.s_only_count}")
        print(f"rows transferred to host:       {one_counter.rows_transferred}")
        assert one_rd.s_only_count == 500
        # Only scalar aggregates and the bounded display sample leave the database; the 500
        # copies themselves are never transferred.
        assert one_counter.rows_transferred <= 8 + 20, "single-side residual must not be expanded"

        # 4) A failure part-way through must still reclaim every staging table.
        report("mid-query failure cleanup")
        adapter.execute("CREATE TABLE fail_src (id INT, v INT)")
        adapter.execute("INSERT INTO fail_src VALUES (1, 1)")
        fail_td = TableDiff(
            adapter,
            "fail_src",
            "missing_tgt_table",
            on=["id"],
            row_match="minimum_change",
        )
        try:
            fail_td.row_diff()
        except Exception as e:  # noqa: BLE001 - boundary test intentionally triggers any DB error
            print(f"query failed as expected: {type(e).__name__}")
        else:
            raise AssertionError("expected a mid-query failure")
        leftovers = adapter.fetchall(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'sqlmesh_temp'"
        )
        assert leftovers == [], f"temp tables leaked after failure: {leftovers}"

        print("\nAll scalability and boundary experiments passed.")


if __name__ == "__main__":
    main()
