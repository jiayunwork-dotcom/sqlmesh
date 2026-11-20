"""Tests for the duplicate-aware ``minimum_change`` row matching of TableDiff."""

from __future__ import annotations

import itertools
import typing as t

import pandas as pd  # noqa: TID253
import pytest
from sqlglot import exp

from sqlmesh.core.config import DuckDBConnectionConfig
from sqlmesh.core.context import Context
from sqlmesh.core.engine_adapter.mixins import RowDiffMixin
from sqlmesh.core.matching import minimum_change_assignment
from sqlmesh.core.table_diff import (
    DEFAULT_MAX_UNMATCHED_ROWS,
    ROW_MATCH_JOIN,
    ROW_MATCH_MINIMUM_CHANGE,
    TableDiff,
)
from sqlmesh.utils.errors import SQLMeshError

pytestmark = pytest.mark.slow


def _adapter() -> t.Any:
    return DuckDBConnectionConfig().create_engine_adapter()


def _create_pair(
    adapter: t.Any,
    columns_to_types: t.Dict[str, exp.DataType],
    src_rows: t.Sequence[t.Tuple],
    tgt_rows: t.Sequence[t.Tuple],
    src: str = "mc_src",
    tgt: str = "mc_tgt",
) -> None:
    adapter.drop_table(src, exists=True)
    adapter.drop_table(tgt, exists=True)
    adapter.create_table(src, columns_to_types)
    adapter.create_table(tgt, columns_to_types)
    if src_rows:
        adapter.insert_append(src, pd.DataFrame(src_rows, columns=list(columns_to_types)))
    if tgt_rows:
        adapter.insert_append(tgt, pd.DataFrame(tgt_rows, columns=list(columns_to_types)))


def _diff(adapter: t.Any, **kwargs: t.Any) -> t.Any:
    kwargs.setdefault("source", "mc_src")
    kwargs.setdefault("target", "mc_tgt")
    kwargs.setdefault("row_match", ROW_MATCH_MINIMUM_CHANGE)
    return TableDiff(adapter, on=["id"], **kwargs).row_diff()


def _assert_conservation(rd: t.Any) -> None:
    assert rd.source_count == (rd.full_match_count + rd.partial_match_count + rd.s_only_count)
    assert rd.target_count == (rd.full_match_count + rd.partial_match_count + rd.t_only_count)


def test_minimum_change_pure_reordering() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    rows = [(1, 10), (1, 20), (1, 30)]
    _create_pair(adapter, cols, rows, list(reversed(rows)))
    rd = _diff(adapter)
    assert rd.full_match_count == 3
    assert rd.partial_match_count == 0
    assert rd.s_only_count == 0
    assert rd.t_only_count == 0
    assert rd.changed_cell_count == 0
    _assert_conservation(rd)


def test_minimum_change_multiplicity_difference() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(
        adapter,
        cols,
        [(1, 10), (1, 10), (1, 20)],
        [(1, 10), (1, 20), (1, 20), (1, 30)],
    )
    rd = _diff(adapter)
    # one full (10,10), one full (20,20); the extra source 10 pairs with a target 20 as a
    # changed pairing, while the remaining target 20 and target 30 are target-only
    assert rd.full_match_count == 2
    assert rd.partial_match_count == 1
    assert rd.s_only_count == 0
    assert rd.t_only_count == 1
    assert rd.changed_cell_count == 1
    _assert_conservation(rd)


def test_minimum_change_composite_null_key() -> None:
    adapter = _adapter()
    cols = {
        "k1": exp.DataType.build("int"),
        "k2": exp.DataType.build("varchar"),
        "v": exp.DataType.build("int"),
    }
    _create_pair(
        adapter,
        cols,
        [(None, None, 1), (None, "x", 2), (1, None, 3)],
        [(None, None, 1), (None, "x", 9), (1, None, 3)],
    )
    rd = TableDiff(
        adapter, "mc_src", "mc_tgt", on=["k1", "k2"], row_match=ROW_MATCH_MINIMUM_CHANGE
    ).row_diff()
    assert rd.full_match_count == 2
    assert rd.partial_match_count == 1
    assert rd.s_only_count == 0
    assert rd.t_only_count == 0
    assert rd.changed_cell_count == 1
    _assert_conservation(rd)


def test_minimum_change_empty_and_single_side() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(adapter, cols, [], [])
    rd = _diff(adapter)
    assert rd.empty
    assert rd.source_count == 0 and rd.target_count == 0

    _create_pair(adapter, cols, [(1, 1)], [])
    rd = _diff(adapter)
    assert rd.s_only_count == 1
    assert rd.t_only_count == 0
    assert rd.full_match_count == 0
    _assert_conservation(rd)

    _create_pair(adapter, cols, [], [(2, 2)])
    rd = _diff(adapter)
    assert rd.t_only_count == 1
    assert rd.s_only_count == 0
    _assert_conservation(rd)


def test_minimum_change_skipped_columns() -> None:
    adapter = _adapter()
    cols = {
        "id": exp.DataType.build("int"),
        "v": exp.DataType.build("int"),
        "ignored": exp.DataType.build("int"),
    }
    _create_pair(
        adapter,
        cols,
        [(1, 10, 100), (1, 20, 200)],
        [(1, 10, 999), (1, 20, 888)],
    )
    rd = TableDiff(adapter, "mc_src", "mc_tgt", on=["id"], skip_columns=["ignored"]).row_diff()
    assert rd.full_match_count == 2
    assert rd.changed_cell_count == 0

    # key columns cannot be skipped
    with pytest.raises(SQLMeshError, match="cannot be skipped"):
        TableDiff(
            adapter,
            "mc_src",
            "mc_tgt",
            on=["id"],
            skip_columns=["id"],
            row_match=ROW_MATCH_MINIMUM_CHANGE,
        ).row_diff()


def test_minimum_change_where_filter_changes_multiplicity() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(
        adapter,
        cols,
        [(1, 10), (1, 10), (1, 20)],
        [(1, 10), (1, 20), (1, 20)],
    )
    rd = TableDiff(
        adapter,
        "mc_src",
        "mc_tgt",
        on=["id"],
        where="v >= 20",
        row_match=ROW_MATCH_MINIMUM_CHANGE,
    ).row_diff()
    # after filtering: source has one 20, target has two 20 -> one full, one target-only
    assert rd.full_match_count == 1
    assert rd.t_only_count == 1
    _assert_conservation(rd)


def test_minimum_change_decimal_precision_boundary() -> None:
    adapter = _adapter()
    cols = {
        "id": exp.DataType.build("int"),
        "v": exp.DataType.build("double"),
    }
    _create_pair(
        adapter,
        cols,
        # chosen so DuckDB's truncation to DECIMAL(38, 3) (via DOUBLE, see DuckDBEngineAdapter)
        # yields equal values at 3 decimals but distinct ones at 4 decimals
        [(1, 1.00049), (2, 2.50049)],
        [(1, 1.00039), (2, 2.50039)],
    )
    rd = TableDiff(
        adapter, "mc_src", "mc_tgt", on=["id"], decimals=3, row_match=ROW_MATCH_MINIMUM_CHANGE
    ).row_diff()
    # at 3 decimals both truncate to 1.000 / 2.500 -> full matches
    assert rd.full_match_count == 2
    rd4 = TableDiff(
        adapter, "mc_src", "mc_tgt", on=["id"], decimals=4, row_match=ROW_MATCH_MINIMUM_CHANGE
    ).row_diff()
    assert rd4.full_match_count == 0
    assert rd4.partial_match_count == 2
    assert rd4.changed_cell_count == 2
    _assert_conservation(rd4)


def test_minimum_change_beats_greedy_pairing() -> None:
    """A greedy closest-row pairing must not be mistaken for the global optimum."""
    adapter = _adapter()
    cols = {
        "id": exp.DataType.build("int"),
        "a": exp.DataType.build("int"),
        "b": exp.DataType.build("int"),
    }
    # source rows: S0=(0,0), S1=(0,9); target rows: T0=(0,9), T1=(9,9)
    # greedy S0->T0 costs 1, then S1->T1 costs 2 => 3; optimum S0->T1 (2), S1->T0 (0) => 2
    _create_pair(
        adapter,
        cols,
        [(1, 0, 0), (1, 0, 9)],
        [(1, 0, 9), (1, 9, 9)],
    )
    rd = _diff(adapter)
    # S1=(0,9) and T0=(0,9) are fully equal first; the remaining S0=(0,0)/T1=(9,9) pair changes
    # both columns. A greedy pass over raw rows could instead pick S0->T0 (1) and S1->T1 (2),
    # totaling 3 changed cells; the exact optimum must still be 2.
    assert rd.full_match_count == 1
    assert rd.partial_match_count == 1
    assert rd.changed_cell_count == 2
    _assert_conservation(rd)


def test_minimum_change_no_comparable_non_key_columns() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int")}
    _create_pair(adapter, cols, [(1,), (1,)], [(1,), (1,), (1,)])
    rd = _diff(adapter)
    # without non-key columns every same-key copy is considered equal
    assert rd.full_match_count == 2
    assert rd.t_only_count == 1
    assert rd.column_stats.shape[0] == 0
    _assert_conservation(rd)


def test_minimum_change_accepted_scalar_types() -> None:
    adapter = _adapter()
    adapter.execute(
        """
        CREATE TABLE mc_src (
            id INT, i BIGINT, b BOOLEAN, f DOUBLE, dc DECIMAL(10, 2),
            txt TEXT, d DATE, ts TIMESTAMP
        )
        """
    )
    adapter.execute(
        """
        CREATE TABLE mc_tgt (
            id INT, i BIGINT, b BOOLEAN, f DOUBLE, dc DECIMAL(10, 2),
            txt TEXT, d DATE, ts TIMESTAMP
        )
        """
    )
    adapter.execute(
        """
        INSERT INTO mc_src VALUES
            (1, 7, TRUE, 1.5, 2.50, 'a,b', DATE '2024-01-02', TIMESTAMP '2024-01-02 03:04:05'),
            (2, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
            (3, 9, FALSE, 3.25, 9.75, '', DATE '2020-12-31', TIMESTAMP '2020-12-31 00:00:00')
        """
    )
    adapter.execute(
        """
        INSERT INTO mc_tgt VALUES
            (1, 7, TRUE, 1.5, 2.50, 'a,b', DATE '2024-01-02', TIMESTAMP '2024-01-02 03:04:05'),
            (2, NULL, NULL, NULL, NULL, '', NULL, NULL),
            (3, 9, TRUE, 3.25, 9.75, 'x', DATE '2020-12-31', TIMESTAMP '2020-12-31 00:00:00')
        """
    )
    rd = _diff(adapter)
    assert rd.full_match_count == 1
    assert rd.partial_match_count == 2
    # row 2: empty string differs from NULL, plus b/d/ts/f/dc -> changed cells counted
    assert rd.changed_cell_count >= 3
    _assert_conservation(rd)


def test_minimum_change_empty_string_not_equal_to_null() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("varchar")}
    _create_pair(adapter, cols, [(1, "")], [(1, None)])
    rd = _diff(adapter)
    assert rd.full_match_count == 0
    assert rd.partial_match_count == 1
    assert rd.changed_cell_count == 1


def test_minimum_change_composite_key_delimiter_collision() -> None:
    """Values containing the key delimiter must not merge distinct keys."""
    adapter = _adapter()
    cols = {
        "k1": exp.DataType.build("varchar"),
        "k2": exp.DataType.build("varchar"),
        "v": exp.DataType.build("int"),
    }
    _create_pair(
        adapter,
        cols,
        [("a", "b,c", 1), ("a,b", "c", 2)],
        [("a", "b,c", 1), ("a,b", "c", 2)],
    )
    rd = TableDiff(
        adapter, "mc_src", "mc_tgt", on=["k1", "k2"], row_match=ROW_MATCH_MINIMUM_CHANGE
    ).row_diff()
    assert rd.full_match_count == 2
    assert rd.s_only_count == 0 and rd.t_only_count == 0


def test_minimum_change_rejects_expression_on() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(adapter, cols, [(1, 1)], [(1, 1)])
    with pytest.raises(SQLMeshError, match="non-empty list of column names"):
        TableDiff(
            adapter,
            "mc_src",
            "mc_tgt",
            on=exp.condition("s.id = t.id"),
            row_match=ROW_MATCH_MINIMUM_CHANGE,
        ).row_diff()
    with pytest.raises(SQLMeshError, match="at least one business key"):
        TableDiff(adapter, "mc_src", "mc_tgt", on=[], row_match=ROW_MATCH_MINIMUM_CHANGE).row_diff()


def test_minimum_change_rejects_unknown_and_mismatched_columns() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(adapter, cols, [(1, 1)], [(1, 1)])
    with pytest.raises(SQLMeshError, match="Invalid business key"):
        TableDiff(
            adapter, "mc_src", "mc_tgt", on=["missing"], row_match=ROW_MATCH_MINIMUM_CHANGE
        ).row_diff()

    adapter.execute("CREATE TABLE mc_other (id INT, v VARCHAR)")
    adapter.execute("INSERT INTO mc_other VALUES (1, '1')")
    with pytest.raises(SQLMeshError, match="type mismatch"):
        TableDiff(
            adapter, "mc_src", "mc_other", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE
        ).row_diff()


def test_minimum_change_rejects_unsupported_type() -> None:
    adapter = _adapter()
    adapter.execute("CREATE TABLE mc_src (id INT, v INTEGER[])")
    adapter.execute("CREATE TABLE mc_tgt (id INT, v INTEGER[])")
    adapter.execute("INSERT INTO mc_src VALUES (1, [1, 2])")
    adapter.execute("INSERT INTO mc_tgt VALUES (1, [1, 2])")
    with pytest.raises(SQLMeshError, match="unsupported type"):
        TableDiff(
            adapter, "mc_src", "mc_tgt", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE
        ).row_diff()


def test_minimum_change_cap_aborts_two_sided_key() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    # 130 distinct unmatched rows on both sides exceeds the default cap of 128
    _create_pair(
        adapter,
        cols,
        [(1, i) for i in range(130)],
        [(1, i + 1000) for i in range(130)],
    )
    with pytest.raises(SQLMeshError, match="max_unmatched_rows=128"):
        _diff(adapter)
    # a raised cap solves it exactly
    rd = TableDiff(
        adapter,
        "mc_src",
        "mc_tgt",
        on=["id"],
        row_match=ROW_MATCH_MINIMUM_CHANGE,
        max_unmatched_rows=130,
    ).row_diff()
    assert rd.partial_match_count == 130
    _assert_conservation(rd)


def test_minimum_change_single_side_large_multiplicity_allowed() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    # only the source side has a huge multiplicity; no pairing is attempted
    _create_pair(adapter, cols, [(1, i) for i in range(300)], [])
    rd = TableDiff(
        adapter,
        "mc_src",
        "mc_tgt",
        on=["id"],
        row_match=ROW_MATCH_MINIMUM_CHANGE,
        max_unmatched_rows=4,
    ).row_diff()
    assert rd.s_only_count == 300
    assert rd.t_only_count == 0
    _assert_conservation(rd)


def test_minimum_change_invalid_max_unmatched_rows() -> None:
    adapter = _adapter()
    for bad in (0, -1, True, "10", 1.5):
        with pytest.raises(ValueError, match="max_unmatched_rows"):
            TableDiff(
                adapter,
                "mc_src",
                "mc_tgt",
                on=["id"],
                row_match=ROW_MATCH_MINIMUM_CHANGE,
                max_unmatched_rows=bad,
            )


def test_minimum_change_invalid_row_match_value() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="Unknown row_match"):
        TableDiff(adapter, "mc_src", "mc_tgt", on=["id"], row_match="bogus")


def test_minimum_change_sample_limit_does_not_change_stats() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(
        adapter,
        cols,
        [(1, i) for i in range(20)],
        [(1, i + 100) for i in range(20)],
    )
    rd_small = TableDiff(
        adapter, "mc_src", "mc_tgt", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE, limit=3
    ).row_diff()
    rd_big = TableDiff(
        adapter, "mc_src", "mc_tgt", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE, limit=50
    ).row_diff()
    assert rd_small.stats == rd_big.stats
    assert rd_small.partial_match_count == 20
    assert len(rd_small.sample) <= 3


def test_minimum_change_result_is_cached_without_rescan() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(adapter, cols, [(1, 1)], [(1, 2)])
    td = TableDiff(adapter, "mc_src", "mc_tgt", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE)
    first = td.row_diff()
    second = td.row_diff()
    assert first is second
    assert first.stats == second.stats


def test_minimum_change_temp_tables_are_cleaned_up() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(adapter, cols, [(1, 1)], [(1, 2)])
    TableDiff(adapter, "mc_src", "mc_tgt", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE).row_diff()
    leftovers = adapter.fetchall(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'sqlmesh_temp'"
    )
    assert leftovers == []


def test_minimum_change_default_row_match_is_join() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(adapter, cols, [(1, 1), (1, 2)], [(1, 1), (1, 2)])
    rd = TableDiff(adapter, "mc_src", "mc_tgt", on=["id"]).row_diff()
    assert rd.row_match == ROW_MATCH_JOIN
    # join mode inflates duplicates: 2 source x 2 target = 4 joined rows
    assert rd.join_count == 4


def test_minimum_change_brute_force_truth() -> None:
    """Enumerate every feasible assignment and verify cost and conservation for small keys."""
    import random

    adapter = _adapter()
    random.seed(42)
    cols = {
        "id": exp.DataType.build("int"),
        "a": exp.DataType.build("int"),
        "b": exp.DataType.build("int"),
    }
    for trial in range(25):
        n_s = random.randint(1, 5)
        n_t = random.randint(1, 5)
        src_rows = [(1, random.randint(0, 3), random.randint(0, 3)) for _ in range(n_s)]
        tgt_rows = [(1, random.randint(0, 3), random.randint(0, 3)) for _ in range(n_t)]
        _create_pair(adapter, cols, src_rows, tgt_rows)
        rd = _diff(adapter)

        def cost(s: t.Tuple, t: t.Tuple) -> int:
            return int(s[1] != t[1]) + int(s[2] != t[2])

        k = min(n_s, n_t)
        best = None
        for s_idx in itertools.combinations(range(n_s), k):
            for t_perm in itertools.permutations(range(n_t), k):
                total = sum(cost(src_rows[i], tgt_rows[j]) for i, j in zip(s_idx, t_perm))
                if best is None or total < best:
                    best = total
        assert rd.changed_cell_count == best
        _assert_conservation(rd)


def test_minimum_change_assignment_tie_break_is_stable() -> None:
    # every pairing has equal cost; the documented rule picks the lexicographically smallest one
    costs = [[0, 0], [0, 0]]
    pairs, total = minimum_change_assignment(costs)
    assert pairs == [(0, 0), (1, 1)]
    assert total == 0
    # physical order independence: feeding rows in a different order cannot change the optimum
    shuffled = [list(reversed(row)) for row in reversed(costs)]
    pairs_2, _ = minimum_change_assignment(shuffled)
    assert pairs_2 == [(0, 0), (1, 1)]


def test_minimum_change_capability_flag_defaults_off() -> None:
    # the base mixin declares no engine support; DuckDB opts in explicitly
    assert RowDiffMixin.SUPPORTS_MINIMUM_CHANGE_TABLE_DIFF is False
    assert (
        DuckDBConnectionConfig().create_engine_adapter().SUPPORTS_MINIMUM_CHANGE_TABLE_DIFF is True
    )
    assert DEFAULT_MAX_UNMATCHED_ROWS == 128


def test_minimum_change_unsupported_engine_is_rejected(mocker: t.Any) -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    _create_pair(adapter, cols, [(1, 1)], [(1, 1)])
    mocker.patch.object(type(adapter), "SUPPORTS_MINIMUM_CHANGE_TABLE_DIFF", False)
    with pytest.raises(SQLMeshError, match="does not support row_match='minimum_change'"):
        TableDiff(
            adapter, "mc_src", "mc_tgt", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE
        ).row_diff()


def test_minimum_change_rejects_non_finite_floats() -> None:
    adapter = _adapter()
    adapter.execute("CREATE TABLE f_src (id INT, v DOUBLE); CREATE TABLE f_tgt (id INT, v DOUBLE)")
    adapter.execute("INSERT INTO f_src VALUES (1, 1.0), (2, 'inf')")
    adapter.execute("INSERT INTO f_tgt VALUES (1, 1.0), (2, 1.0)")
    with pytest.raises(SQLMeshError, match="finite float/DECIMAL"):
        TableDiff(
            adapter, "f_src", "f_tgt", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE
        ).row_diff()
    # filtering the non-finite rows out makes the diff valid again
    rd = TableDiff(
        adapter,
        "f_src",
        "f_tgt",
        on=["id"],
        where="id = 1",
        row_match=ROW_MATCH_MINIMUM_CHANGE,
    ).row_diff()
    assert rd.full_match_count == 1
    _assert_conservation(rd)


def test_minimum_change_weighted_sample_merges_identical_pairings() -> None:
    from io import StringIO

    from rich.console import Console

    from sqlmesh.core.console import TerminalConsole

    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    # three identical changed pairings plus one fully equal row on each side
    _create_pair(
        adapter,
        cols,
        [(1, 10), (2, 1), (2, 1), (2, 1)],
        [(1, 10), (2, 9), (2, 9), (2, 9)],
    )
    rd = _diff(adapter, limit=20)
    assert rd.full_match_count == 1
    assert rd.partial_match_count == 3
    assert rd.changed_cell_count == 3
    # every detail sample row carries its own multiplicity
    changed = rd.joined_sample
    assert len(changed) == 3
    assert changed["__sqlmesh_multiplicity"].tolist() == [1, 1, 1]
    # identical pairings are merged for display without losing the total weight
    output = StringIO()
    TerminalConsole(console=Console(file=output, force_terminal=False, width=120)).show_row_diff(
        rd, show_sample=True
    )
    rendered = output.getvalue()
    assert "MULTIPLICITY" in rendered
    assert "3" in rendered
    _assert_conservation(rd)


def test_minimum_change_query_failure_cleans_temp() -> None:
    adapter = _adapter()
    cols = {"id": exp.DataType.build("int"), "v": exp.DataType.build("int")}
    adapter.drop_table("mc_src", exists=True)
    adapter.drop_table("mc_tgt", exists=True)
    adapter.create_table("mc_src", cols)
    adapter.create_table("mc_tgt", cols)
    adapter.execute("INSERT INTO mc_src VALUES (1, 1)")
    adapter.execute("INSERT INTO mc_tgt VALUES (1, 1)")
    adapter.drop_table("mc_tgt")
    with pytest.raises(Exception):
        TableDiff(
            adapter, "mc_src", "mc_tgt", on=["id"], row_match=ROW_MATCH_MINIMUM_CHANGE
        ).row_diff()
    leftovers = adapter.fetchall(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'sqlmesh_temp'"
    )
    assert leftovers == []


def test_minimum_change_through_context_model_environments(sushi_context_fixed_date: t.Any) -> None:
    """The full Context path resolves physical tables and the model grain itself."""
    from sqlmesh.core import dialect as d
    from sqlmesh.core.config import CategorizerConfig, AutoCategorizationMode
    from sqlmesh.core.model import SqlModel, load_sql_based_model

    context = sushi_context_fixed_date
    expressions = d.parse(
        """
        MODEL (name memory.sushi.mc_items, kind full, grain (key_1, key_2));
        SELECT key_1, key_2, value
        FROM (VALUES
            (1, 'a', 10),
            (1, 'a', 10),
            (1, 'a', 20),
            (2, 'b', 30),
        ) AS t (key_1, key_2, value)
        """
    )
    context.upsert_model(load_sql_based_model(expressions, dialect="snowflake"))
    context.plan(
        "mc_source",
        no_prompts=True,
        auto_apply=True,
        skip_tests=True,
        start="2023-01-31",
        end="2023-01-31",
    )

    model = context.models['"MEMORY"."SUSHI"."MC_ITEMS"']
    modified = model.dict()
    modified["query"] = (
        exp.select("*")
        .from_(model.query.subquery())
        .union(
            "SELECT key_1, key_2, value FROM (VALUES "
            "(1, 'a', 99), (3, 'c', 40)) AS t (key_1, key_2, value)"
        )
    )
    context.upsert_model(SqlModel(**modified))
    context.auto_categorize_changes = CategorizerConfig(sql=AutoCategorizationMode.FULL)
    context.plan(
        "mc_target",
        create_from="mc_source",
        no_prompts=True,
        auto_apply=True,
        skip_tests=True,
        start="2023-01-31",
        end="2023-01-31",
    )

    # grain (from the model) is used without passing `on`
    diffs = context.table_diff(
        source="mc_source",
        target="mc_target",
        select_models={"memory.sushi.mc*"},
        row_match=ROW_MATCH_MINIMUM_CHANGE,
        show=False,
    )
    assert len(diffs) == 1
    table_diff = diffs[0]
    # the environment-specific physical views were resolved
    assert table_diff.source != table_diff.target
    rd = table_diff.row_diff()
    assert rd.is_minimum_change
    # the two (10) copies and the (30) copy fully match; the source (20) is paired with the
    # changed target (99) as one changed pairing, while (40) is target-only
    assert rd.full_match_count == 3
    assert rd.partial_match_count == 1
    assert rd.s_only_count == 0
    assert rd.t_only_count == 1
    assert rd.changed_cell_count == 1
    _assert_conservation(rd)


def test_minimum_change_through_cli(tmp_path: t.Any) -> None:
    """The real CLI flag reaches the Context and renders the weighted output."""
    from pathlib import Path

    from click.testing import CliRunner

    from sqlmesh.cli.main import cli
    from sqlmesh.core.config import (
        Config,
        DuckDBConnectionConfig,
        GatewayConfig,
        ModelDefaultsConfig,
    )

    # A file-backed database so the CLI's own Context process attaches the same catalog.
    db_path = str(Path(tmp_path) / "cli_diff.db")
    (Path(tmp_path) / "config.yaml").write_text(
        f"""gateways:
  local:
    connection:
      type: duckdb
      database: {db_path}

default_gateway: local

model_defaults:
  dialect: duckdb
"""
    )
    connection = DuckDBConnectionConfig(database=db_path)
    config = Config(
        project="mc_cli_project",
        gateways={"gw": GatewayConfig(connection=connection)},
        model_defaults=ModelDefaultsConfig(dialect="duckdb"),
    )
    context = Context(paths=tmp_path, config=config)
    # The CLI requires a project layout containing at least one model.
    models_dir = Path(tmp_path) / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "placeholder.sql").write_text("MODEL (name placeholder); SELECT 1 AS c")
    adapter = context.engine_adapter
    adapter.execute(
        "CREATE TABLE mc_cli_src_table (key_1 INT, value INT); "
        "INSERT INTO mc_cli_src_table VALUES (1, 10), (1, 10), (1, 20)"
    )
    adapter.execute(
        "CREATE TABLE mc_cli_tgt_table (key_1 INT, value INT); "
        "INSERT INTO mc_cli_tgt_table VALUES (1, 10), (1, 10), (1, 99)"
    )

    result = CliRunner().invoke(
        cli,
        [
            "--paths",
            str(tmp_path),
            "table_diff",
            "mc_cli_src_table:mc_cli_tgt_table",
            "-o",
            "key_1",
            "--row-match",
            "minimum-change",
            "--show-sample",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "MINIMUM CHANGE" in result.output
    assert "TOTAL CHANGED CELLS" in result.output
    assert "MULTIPLICITY" in result.output
