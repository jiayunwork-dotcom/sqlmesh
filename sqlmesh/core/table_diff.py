from __future__ import annotations

import math
import typing as t
from functools import cached_property

from sqlmesh.core.dialect import to_schema
from sqlmesh.core.engine_adapter.mixins import RowDiffMixin
from sqlmesh.core.engine_adapter.athena import AthenaEngineAdapter
from sqlmesh.core.matching import minimum_change_assignment
from sqlglot import exp, parse_one
from sqlglot.helper import ensure_list
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify_columns import quote_identifiers
from sqlglot.optimizer.scope import find_all_in_scope

from sqlmesh.utils.pydantic import PydanticModel
from sqlmesh.utils.errors import SQLMeshError


if t.TYPE_CHECKING:
    import pandas as pd

    from sqlmesh.core._typing import TableName
    from sqlmesh.core.engine_adapter import EngineAdapter

SQLMESH_JOIN_KEY_COL = "__sqlmesh_join_key"
SQLMESH_SAMPLE_TYPE_COL = "__sqlmesh_sample_type"
SQLMESH_SIGNATURE_COL = "__sqlmesh_row_signature"
SQLMESH_ROW_NUMBER_COL = "__sqlmesh_row_number"
SQLMESH_CLASS_ORDINAL_COL = "__sqlmesh_class_ordinal"
SQLMESH_MULTIPLICITY_COL = "__sqlmesh_multiplicity"

# Row matching strategies accepted by TableDiff / Context.table_diff.
ROW_MATCH_JOIN = "join"
ROW_MATCH_MINIMUM_CHANGE = "minimum_change"
ROW_MATCH_VALUES = (ROW_MATCH_JOIN, ROW_MATCH_MINIMUM_CHANGE)

# Default cap on unmatched copies per side within a single business key. See the
# ``max_unmatched_rows`` argument of :class:`TableDiff` for details.
DEFAULT_MAX_UNMATCHED_ROWS = 128


class SchemaDiff(PydanticModel, frozen=True):
    """An object containing the schema difference between a source and target table."""

    source: str
    target: str
    source_schema: t.Dict[str, exp.DataType]
    target_schema: t.Dict[str, exp.DataType]
    source_alias: t.Optional[str] = None
    target_alias: t.Optional[str] = None
    model_name: t.Optional[str] = None
    ignore_case: bool = False

    @property
    def _comparable_source_schema(self) -> t.Dict[str, exp.DataType]:
        return (
            self._lowercase_schema_names(self.source_schema)
            if self.ignore_case
            else self.source_schema
        )

    @property
    def _comparable_target_schema(self) -> t.Dict[str, exp.DataType]:
        return (
            self._lowercase_schema_names(self.target_schema)
            if self.ignore_case
            else self.target_schema
        )

    def _lowercase_schema_names(
        self, schema: t.Dict[str, exp.DataType]
    ) -> t.Dict[str, exp.DataType]:
        return {c.lower(): t for c, t in schema.items()}

    def _original_column_name(
        self, maybe_lowercased_column_name: str, schema: t.Dict[str, exp.DataType]
    ) -> str:
        if not self.ignore_case:
            return maybe_lowercased_column_name

        return next(c for c in schema if c.lower() == maybe_lowercased_column_name)

    @property
    def added(self) -> t.List[t.Tuple[str, exp.DataType]]:
        """Added columns."""
        return [
            (self._original_column_name(c, self.target_schema), t)
            for c, t in self._comparable_target_schema.items()
            if c not in self._comparable_source_schema
        ]

    @property
    def removed(self) -> t.List[t.Tuple[str, exp.DataType]]:
        """Removed columns."""
        return [
            (self._original_column_name(c, self.source_schema), t)
            for c, t in self._comparable_source_schema.items()
            if c not in self._comparable_target_schema
        ]

    @property
    def modified(self) -> t.Dict[str, t.Tuple[exp.DataType, exp.DataType]]:
        """Columns with modified types."""
        modified = {}
        for column in self._comparable_source_schema.keys() & self._comparable_target_schema.keys():
            source_type = self._comparable_source_schema[column]
            target_type = self._comparable_target_schema[column]

            if source_type != target_type:
                modified[column] = (source_type, target_type)

        if self.ignore_case:
            modified = {
                self._original_column_name(c, self.source_schema): dt for c, dt in modified.items()
            }

        return modified

    @property
    def has_changes(self) -> bool:
        """Does the schema contain any changes at all between source and target"""
        return bool(self.added or self.removed or self.modified)


class RowDiff(PydanticModel, frozen=True):
    """Summary statistics and a sample dataframe."""

    source: str
    target: str
    stats: t.Dict[str, float]
    sample: pd.DataFrame
    joined_sample: pd.DataFrame
    s_sample: pd.DataFrame
    t_sample: pd.DataFrame
    column_stats: pd.DataFrame
    source_alias: t.Optional[str] = None
    target_alias: t.Optional[str] = None
    model_name: t.Optional[str] = None
    decimals: int = 3
    # The row matching strategy that produced this diff. ``join`` is the historical
    # cross-join-on-grain behavior; ``minimum_change`` is the bounded duplicate-aware pairing.
    row_match: str = ROW_MATCH_JOIN

    _types_resolved: t.ClassVar[bool] = False

    def __new__(cls, *args: t.Any, **kwargs: t.Any) -> RowDiff:
        if not cls._types_resolved:
            cls._resolve_types()
        return super().__new__(cls)

    @classmethod
    def _resolve_types(cls) -> None:
        # Pandas is imported by type checking so we need to resolve the types with the real import before instantiating
        import pandas as pd  # noqa

        cls.model_rebuild()
        cls._types_resolved = True

    @property
    def source_count(self) -> int:
        """Count of the source."""
        return int(self.stats["s_count"])

    @property
    def target_count(self) -> int:
        """Count of the target."""
        return int(self.stats["t_count"])

    @property
    def empty(self) -> bool:
        return (
            self.source_count == 0
            and self.target_count == 0
            and self.s_only_count == 0
            and self.t_only_count == 0
        )

    @property
    def count_pct_change(self) -> float:
        """The percentage change of the counts."""
        if self.source_count == 0:
            return math.inf
        return ((self.target_count - self.source_count) / self.source_count) * 100

    @property
    def join_count(self) -> int:
        """Count of successfully joined rows."""
        return int(self.stats["join_count"])

    @property
    def is_minimum_change(self) -> bool:
        """Whether this diff was produced by the duplicate-aware minimum-change pairing."""
        return self.row_match == ROW_MATCH_MINIMUM_CHANGE

    @property
    def full_match_count(self) -> int:
        """The number of rows for which shared columns have same values."""
        return int(self.stats["full_match_count"])

    @property
    def full_match_pct(self) -> float:
        """The percentage of rows for which shared columns have same values."""
        return self._pct(2 * self.full_match_count)

    @property
    def partial_match_count(self) -> int:
        """The number of rows for which some shared columns have same values."""
        return self.join_count - self.full_match_count

    @property
    def partial_match_pct(self) -> float:
        """The percentage of rows for which some shared columns have same values."""
        return self._pct(2 * self.partial_match_count)

    @property
    def changed_cell_count(self) -> int:
        """Total number of differing non-key cells across the changed pairings.

        Only populated by the ``minimum_change`` row matching strategy; the historical join
        mode reports ``0`` because its cross-joined rows do not represent a single pairing.
        """
        return int(self.stats.get("changed_cell_count", 0))

    @property
    def s_only_count(self) -> int:
        """Count of rows only present in source."""
        return int(self.stats["s_only_count"])

    @property
    def s_only_pct(self) -> float:
        """The percentage of rows that are only present in source."""
        return self._pct(self.s_only_count)

    @property
    def t_only_count(self) -> int:
        """Count of rows only present in target."""
        return int(self.stats["t_only_count"])

    @property
    def t_only_pct(self) -> float:
        """The percentage of rows that are only present in target."""
        return self._pct(self.t_only_count)

    def _pct(self, numerator: int) -> float:
        return round((numerator / (self.source_count + self.target_count)) * 100, 2)


class TableDiff:
    """Calculates differences between tables, taking into account schema and row level differences."""

    def __init__(
        self,
        adapter: EngineAdapter,
        source: TableName,
        target: TableName,
        on: t.List[str] | exp.Expr,
        skip_columns: t.List[str] | None = None,
        where: t.Optional[str | exp.Expr] = None,
        limit: int = 20,
        source_alias: t.Optional[str] = None,
        target_alias: t.Optional[str] = None,
        model_name: t.Optional[str] = None,
        model_dialect: t.Optional[str] = None,
        decimals: int = 3,
        schema_diff_ignore_case: bool = False,
        row_match: str = ROW_MATCH_JOIN,
        max_unmatched_rows: int = DEFAULT_MAX_UNMATCHED_ROWS,
    ):
        if not isinstance(adapter, RowDiffMixin):
            raise ValueError(f"Engine {adapter} doesnt support RowDiff")

        if row_match not in ROW_MATCH_VALUES:
            raise ValueError(
                f"Unknown row_match '{row_match}'. Expected one of: {', '.join(ROW_MATCH_VALUES)}."
            )

        if isinstance(max_unmatched_rows, bool) or not isinstance(max_unmatched_rows, int):
            raise ValueError(
                "max_unmatched_rows must be a positive integer, got "
                f"{type(max_unmatched_rows).__name__}: {max_unmatched_rows!r}."
            )
        if max_unmatched_rows <= 0:
            raise ValueError(
                f"max_unmatched_rows must be a positive integer, got {max_unmatched_rows}."
            )

        self.adapter = adapter
        self.source = source
        self.target = target
        self.dialect = adapter.dialect
        self.source_table = exp.to_table(self.source, dialect=self.dialect)
        self.target_table = exp.to_table(self.target, dialect=self.dialect)
        self.where = exp.condition(where, dialect=self.dialect) if where else None
        self.limit = limit
        self.model_name = model_name
        self.model_dialect = model_dialect
        self.decimals = decimals
        self.schema_diff_ignore_case = schema_diff_ignore_case
        self.row_match = row_match
        self.max_unmatched_rows = max_unmatched_rows

        # Support environment aliases for diff output improvement in certain cases
        self.source_alias = source_alias
        self.target_alias = target_alias

        cols: t.List[str] = ensure_list(skip_columns)
        self.skip_columns = {
            normalize_identifiers(
                exp.parse_identifier(col),
                dialect=self.model_dialect or self.dialect,
            ).name
            for col in cols
        }

        self._on = on
        self._row_diff: t.Optional[RowDiff] = None

    @property
    def is_minimum_change(self) -> bool:
        return self.row_match == ROW_MATCH_MINIMUM_CHANGE

    @cached_property
    def source_schema(self) -> t.Dict[str, exp.DataType]:
        return self.adapter.columns(self.source_table)

    @cached_property
    def target_schema(self) -> t.Dict[str, exp.DataType]:
        return self.adapter.columns(self.target_table)

    @cached_property
    def key_columns(self) -> t.Tuple[t.List[exp.Column], t.List[exp.Column], t.List[str]]:
        dialect = self.model_dialect or self.dialect

        # If the columns to join on are explicitly specified, then just return them
        if isinstance(self._on, (list, tuple)):
            identifiers = [normalize_identifiers(c, dialect=dialect) for c in self._on]
            s_index = [exp.column(c, "s") for c in identifiers]
            t_index = [exp.column(c, "t") for c in identifiers]
            return s_index, t_index, [i.name for i in identifiers]

        # Otherwise, we need to parse them out of the supplied "on" condition
        index_cols = []
        s_index = []
        t_index = []

        normalize_identifiers(self._on, dialect=dialect)
        for col in self._on.find_all(exp.Column):
            index_cols.append(col.name)
            if col.table.lower() == "s":
                s_index.append(col)
            elif col.table.lower() == "t":
                t_index.append(col)

        index_cols = list(dict.fromkeys(index_cols))
        s_index = list(dict.fromkeys(s_index))
        t_index = list(dict.fromkeys(t_index))

        return s_index, t_index, index_cols

    def _minimum_change_key_columns(self) -> t.List[str]:
        """Resolve and validate the full business key for the minimum-change comparison.

        Only a non-empty list of plain column names is accepted: arbitrary join expressions
        cannot be interpreted as a complete, null-aware business key and are rejected before
        any business data is scanned.
        """
        if not isinstance(self._on, (list, tuple)):
            raise SQLMeshError(
                "row_match='minimum_change' requires 'on' to be a non-empty list of column "
                f"names (for example on=['id', 'event_date']), got join expression: {self._on.sql(dialect=self.model_dialect or self.dialect)}"
            )
        if not self._on:
            raise SQLMeshError(
                "row_match='minimum_change' requires at least one business key column in 'on'."
            )

        dialect = self.model_dialect or self.dialect
        key_names: t.List[str] = []
        for col in self._on:
            parsed = exp.parse_identifier(col) if isinstance(col, str) else col
            identifier = normalize_identifiers(parsed, dialect=dialect)
            if not isinstance(identifier, exp.Identifier) or identifier.name == "":
                raise SQLMeshError(
                    f"row_match='minimum_change' only accepts plain column names in 'on', got: {col!r}."
                )
            if identifier.name not in key_names:
                key_names.append(identifier.name)
        return key_names

    def _validate_minimum_change_inputs(
        self, key_names: t.List[str]
    ) -> t.Tuple[t.Dict[str, exp.DataType], t.Dict[str, exp.DataType], t.Dict[str, exp.DataType]]:
        """Run every pre-scan validation for the minimum-change comparison.

        Returns the filtered source schema, filtered target schema and the mapping of columns
        that can participate in cell-level comparison (same name, same supported type on both
        sides). Raises ``SQLMeshError`` with a clear message instead of silently falling back to
        string comparison or to the historical join behavior.
        """
        if not getattr(self.adapter, "SUPPORTS_MINIMUM_CHANGE_TABLE_DIFF", False):
            raise SQLMeshError(
                f"Engine '{self.adapter.dialect}' does not support row_match='minimum_change'. "
                "Only DuckDB is supported in the initial implementation; omit row_match to use "
                "the default join-based table diff."
            )

        skipped_keys = [c for c in key_names if c in self.skip_columns]
        if skipped_keys:
            raise SQLMeshError(
                f"Business key columns cannot be skipped: {', '.join(skipped_keys)}. Remove them "
                "from skip_columns."
            )

        source_schema = {c: t for c, t in self.source_schema.items() if c not in self.skip_columns}
        target_schema = {c: t for c, t in self.target_schema.items() if c not in self.skip_columns}

        unknown_source = [c for c in key_names if c not in source_schema]
        unknown_target = [c for c in key_names if c not in target_schema]
        if unknown_source or unknown_target:
            details = []
            if unknown_source:
                details.append(f"missing from source '{self.source}': {', '.join(unknown_source)}")
            if unknown_target:
                details.append(f"missing from target '{self.target}': {', '.join(unknown_target)}")
            raise SQLMeshError(
                "Invalid business key column(s) for row_match='minimum_change' - "
                + "; ".join(details)
                + "."
            )

        # Every key column must be a supported, identically-typed scalar on both sides.
        for col in key_names:
            s_type = source_schema[col]
            t_type = target_schema[col]
            if s_type != t_type:
                raise SQLMeshError(
                    f"Business key column '{col}' has mismatched types between source "
                    f"({s_type.sql()}) and target ({t_type.sql()}); row_match='minimum_change' "
                    "requires identical key column types."
                )
            if not self.adapter.supports_minimum_change_column_type(s_type):
                raise SQLMeshError(
                    f"Business key column '{col}' has unsupported type {s_type.sql()} for "
                    "row_match='minimum_change'. Supported types are integers, booleans, finite "
                    "floats/decimals, text, DATE and TIMESTAMP."
                )

        # Non-key columns participate only when the name and type match exactly and the type is
        # supported. Unsupported or mismatched columns are reported loudly rather than coerced.
        comparable: t.Dict[str, exp.DataType] = {}
        rejected: t.List[str] = []
        for name, s_type in source_schema.items():
            if name not in target_schema:
                continue
            t_type = target_schema[name]
            if s_type != t_type:
                rejected.append(
                    f"{name}: source {s_type.sql()} vs target {t_type.sql()} (type mismatch)"
                )
                continue
            if not self.adapter.supports_minimum_change_column_type(s_type):
                rejected.append(f"{name}: unsupported type {s_type.sql()}")
                continue
            comparable[name] = s_type
        for name, t_type in target_schema.items():
            if name not in source_schema:
                continue

        if rejected:
            raise SQLMeshError(
                "row_match='minimum_change' can only compare same-name, same-type scalar "
                "columns. Resolve or skip the following column(s):\n - " + "\n - ".join(rejected)
            )

        self._validate_finite_floats(comparable)

        return source_schema, target_schema, comparable

    def _validate_finite_floats(self, comparable: t.Dict[str, exp.DataType]) -> None:
        """Reject non-finite float values before staging, with a clear error.

        Infinity and NaN cannot be rounded into a fixed-scale DECIMAL for the normalized row
        signature, and comparing them as raw strings would be meaningless, so they are reported
        instead of producing a database cast error or a silent mismatch.
        """
        real_columns = {
            name: type_
            for name, type_ in comparable.items()
            if type_.is_type(*exp.DataType.REAL_TYPES)
        }
        if not real_columns:
            return

        for table, alias in ((self.source_table, "s"), (self.target_table, "t")):
            non_finite = [
                exp.func(
                    "SUM",
                    exp.case()
                    .when(
                        exp.func("ISFINITE", exp.column(name, alias)).not_(),
                        1,
                    )
                    .else_(0),
                )
                for name in real_columns
            ]
            query = exp.select(*non_finite).from_(table.as_(alias))
            if self.where:
                query = query.where(self.where.copy())
            counts = self.adapter.fetchone(query)
            offending = [
                name for name, count in zip(real_columns, counts) if count and int(count) > 0
            ]
            if offending:
                raise SQLMeshError(
                    f"row_match='minimum_change' only supports finite float/DECIMAL values, but "
                    f"table '{table.sql(dialect=self.dialect)}' contains NaN or infinity in "
                    f"column(s): {', '.join(offending)}. Filter out or replace these values before "
                    "diffing."
                )

    @property
    def source_key_expression(self) -> exp.Expr:
        s_index, _, _ = self.key_columns
        return self._key_expression(s_index, self.source_schema)

    @property
    def target_key_expression(self) -> exp.Expr:
        _, t_index, _ = self.key_columns
        return self._key_expression(t_index, self.target_schema)

    def _minimum_change_key_expression(self, key_names: t.List[str], table_alias: str) -> exp.Expr:
        """Collision-free, null-aware encoding of the full business key of one side."""
        if len(key_names) == 1:
            # Keep single-column keys on their native expression so existing indexes are usable.
            # Null-safe equality (IS NOT DISTINCT FROM) is used when matching, never ``=``.
            return exp.column(key_names[0], table_alias)
        columns_to_types = {
            name: (self.source_schema if table_alias == "s" else self.target_schema)[name]
            for name in key_names
        }
        return self.adapter.concat_columns(columns_to_types, self.decimals)

    def _key_expression(
        self, cols: t.List[exp.Column], schema: t.Dict[str, exp.DataType]
    ) -> exp.Expr:
        # if there is a single column, dont do anything fancy to it in order to allow existing indexes to be hit
        if len(cols) == 1:
            return exp.to_column(cols[0].name)

        # if there are multiple columns, turn them into a single column by stringify-ing/concatenating them together
        key_columns_to_types = {key.name: schema[key.name] for key in cols}
        return self.adapter.concat_columns(key_columns_to_types, self.decimals)

    def schema_diff(self) -> SchemaDiff:
        return SchemaDiff(
            source=self.source,
            target=self.target,
            source_schema=self.source_schema,
            target_schema=self.target_schema,
            source_alias=self.source_alias,
            target_alias=self.target_alias,
            model_name=self.model_name,
            ignore_case=self.schema_diff_ignore_case,
        )

    def row_diff(
        self, temp_schema: t.Optional[str] = None, skip_grain_check: bool = False
    ) -> RowDiff:
        if self._row_diff is None:
            if self.is_minimum_change:
                self._row_diff = self._minimum_change_row_diff(temp_schema=temp_schema)
                return self._row_diff

            source_schema = {
                c: t for c, t in self.source_schema.items() if c not in self.skip_columns
            }
            target_schema = {
                c: t for c, t in self.target_schema.items() if c not in self.skip_columns
            }

            s_selects = {c: exp.column(c, "s").as_(f"s__{c}") for c in source_schema}
            t_selects = {c: exp.column(c, "t").as_(f"t__{c}") for c in target_schema}
            s_selects[SQLMESH_JOIN_KEY_COL] = exp.column(SQLMESH_JOIN_KEY_COL, "s").as_(
                f"s__{SQLMESH_JOIN_KEY_COL}"
            )
            t_selects[SQLMESH_JOIN_KEY_COL] = exp.column(SQLMESH_JOIN_KEY_COL, "t").as_(
                f"t__{SQLMESH_JOIN_KEY_COL}"
            )

            matched_columns = {c: t for c, t in source_schema.items() if t == target_schema.get(c)}

            s_index, t_index, index_cols = self.key_columns
            s_index_names = [c.name for c in s_index]
            t_index_names = [t.name for t in t_index]

            def _column_expr(name: str, table: str) -> exp.Expr:
                column_type = matched_columns[name]
                qualified_column = exp.column(name, table)

                if column_type.is_type(*exp.DataType.REAL_TYPES):
                    return self.adapter._normalize_decimal_value(qualified_column, self.decimals)
                if column_type.is_type(*exp.DataType.NESTED_TYPES):
                    return self.adapter._normalize_nested_value(qualified_column)

                return qualified_column

            comparisons = [
                exp.Case()
                .when(_column_expr(c, "s").eq(_column_expr(c, "t")), exp.Literal.number(1))
                .when(
                    exp.column(c, "s").is_(exp.Null()) & exp.column(c, "t").is_(exp.Null()),
                    exp.Literal.number(1),
                )
                .when(
                    exp.column(c, "s").is_(exp.Null()) | exp.column(c, "t").is_(exp.Null()),
                    exp.Literal.number(0),
                )
                .else_(exp.Literal.number(0))
                .as_(f"{c}_matches")
                for c, t in matched_columns.items()
            ]

            source_query = (
                exp.select(
                    *(exp.column(c) for c in source_schema),
                    self.source_key_expression.as_(SQLMESH_JOIN_KEY_COL),
                )
                .from_(self.source_table.as_("s"))
                .where(self.where)
            )
            target_query = (
                exp.select(
                    *(exp.column(c) for c in target_schema),
                    self.target_key_expression.as_(SQLMESH_JOIN_KEY_COL),
                )
                .from_(self.target_table.as_("t"))
                .where(self.where)
            )

            # Ensure every column is qualified with the alias in the source and target queries
            for col in find_all_in_scope(source_query, exp.Column):
                col.set("table", exp.to_identifier("s"))
            for col in find_all_in_scope(target_query, exp.Column):
                col.set("table", exp.to_identifier("t"))

            source_table = exp.table_("__source")
            target_table = exp.table_("__target")
            stats_table = exp.table_("__stats")

            stats_query = (
                exp.select(
                    *s_selects.values(),
                    *t_selects.values(),
                    exp.func(
                        "IF", exp.column(SQLMESH_JOIN_KEY_COL, "s").is_(exp.Null()).not_(), 1, 0
                    ).as_("s_exists"),
                    exp.func(
                        "IF", exp.column(SQLMESH_JOIN_KEY_COL, "t").is_(exp.Null()).not_(), 1, 0
                    ).as_("t_exists"),
                    exp.func(
                        "IF",
                        exp.column(SQLMESH_JOIN_KEY_COL, "s").eq(
                            exp.column(SQLMESH_JOIN_KEY_COL, "t")
                        ),
                        1,
                        0,
                    ).as_("row_joined"),
                    exp.func(
                        "IF",
                        exp.or_(
                            *(
                                exp.and_(
                                    s.is_(exp.Null()),
                                    t.is_(exp.Null()),
                                )
                                for s, t in zip(s_index, t_index)
                            ),
                        ),
                        1,
                        0,
                    ).as_("null_grain"),
                    *comparisons,
                )
                .from_(source_table.as_("s"))
                .join(
                    target_table.as_("t"),
                    on=exp.column(SQLMESH_JOIN_KEY_COL, "s").eq(
                        exp.column(SQLMESH_JOIN_KEY_COL, "t")
                    ),
                    join_type="FULL",
                )
            )

            base_query = (
                exp.Select()
                .with_(source_table, source_query)
                .with_(target_table, target_query)
                .with_(stats_table, stats_query)
                .select(
                    "*",
                    exp.Case()
                    .when(
                        exp.and_(
                            *[
                                exp.column(f"{c}_matches").eq(exp.Literal.number(1))
                                for c in matched_columns
                            ]
                        ),
                        exp.Literal.number(1),
                    )
                    .else_(exp.Literal.number(0))
                    .as_("row_full_match"),
                )
                .from_(stats_table)
            )

            query = self.adapter.ensure_nulls_for_unmatched_after_join(
                quote_identifiers(base_query.copy(), dialect=self.model_dialect or self.dialect)
            )

            if not temp_schema:
                temp_schema = "sqlmesh_temp"

            schema = to_schema(temp_schema, dialect=self.dialect)
            temp_table = exp.table_("diff", db=schema.db, catalog=schema.catalog, quoted=True)

            temp_table_kwargs: t.Dict[str, t.Any] = {}
            if isinstance(self.adapter, AthenaEngineAdapter):
                # Athena has two table formats: Hive (the default) and Iceberg. TableDiff requires that
                # the formats be the same for the source, target, and temp tables.
                source_table_type = self.adapter._query_table_type(self.source_table)
                target_table_type = self.adapter._query_table_type(self.target_table)

                if source_table_type == "iceberg" and target_table_type == "iceberg":
                    temp_table_kwargs["table_format"] = "iceberg"
                # Sets the temp table's format to Iceberg.
                # If neither source nor target table is Iceberg, it defaults to Hive (Athena's default).
                elif source_table_type == "iceberg" or target_table_type == "iceberg":
                    raise SQLMeshError(
                        f"Source table '{self.source}' format '{source_table_type}' and target table '{self.target}' format '{target_table_type}' "
                        f"do not match for Athena. Diffing between different table formats is not supported."
                    )

            with self.adapter.temp_table(
                query, name=temp_table, target_columns_to_types=None, **temp_table_kwargs
            ) as table:
                summary_sums = [
                    exp.func("SUM", "s_exists").as_("s_count"),
                    exp.func("SUM", "t_exists").as_("t_count"),
                    exp.func("SUM", "row_joined").as_("join_count"),
                    exp.func("SUM", "null_grain").as_("null_grain_count"),
                    exp.func("SUM", "row_full_match").as_("full_match_count"),
                    *(exp.func("SUM", name(c)).as_(c.alias) for c in comparisons),
                ]

                if not skip_grain_check:
                    summary_sums.extend(
                        [
                            parse_one(f"COUNT(DISTINCT(s__{SQLMESH_JOIN_KEY_COL}))").as_(
                                "distinct_count_s"
                            ),
                            parse_one(f"COUNT(DISTINCT(t__{SQLMESH_JOIN_KEY_COL}))").as_(
                                "distinct_count_t"
                            ),
                        ]
                    )

                summary_query = exp.select(*summary_sums).from_(table)

                stats_df = self.adapter.fetchdf(summary_query, quote_identifiers=True).fillna(0)
                stats_df["s_only_count"] = stats_df["s_count"] - stats_df["join_count"]
                stats_df["t_only_count"] = stats_df["t_count"] - stats_df["join_count"]
                stats = stats_df.iloc[0].to_dict()

                column_stats_query = (
                    exp.select(
                        *(
                            exp.func(
                                "ROUND",
                                100
                                * (
                                    exp.cast(
                                        exp.func("SUM", name(c)), exp.DataType.build("NUMERIC")
                                    )
                                    / exp.func("COUNT", name(c))
                                ),
                                9,
                            ).as_(c.alias)
                            for c in comparisons
                        )
                    )
                    .from_(table)
                    .where(exp.column("row_joined").eq(exp.Literal.number(1)))
                )

                column_stats = (
                    self.adapter.fetchdf(column_stats_query, quote_identifiers=True)
                    .T.rename(
                        columns={0: "pct_match"},
                        index=lambda x: str(x).replace("_matches", "") if x else "",
                    )
                    # errors=ignore because all the index_cols might not be present in the DF if the `on` condition was something like "s.id == t.item_id"
                    # because these would not be present in the matching_cols (since they have different names) and thus no summary would be generated
                    .drop(index=index_cols, errors="ignore")
                )

                sample = self._fetch_sample(
                    table, s_selects, s_index, t_selects, t_index, self.limit
                )

                joined_sample_cols = [f"s__{c}" for c in s_index_names]
                comparison_cols = [
                    (f"s__{c}", f"t__{c}")
                    for c in column_stats[column_stats["pct_match"] < 100].index
                ]

                for cols in comparison_cols:
                    joined_sample_cols.extend(cols)

                joined_renamed_cols = {
                    c: c.split("__")[1] if c.split("__")[1] in index_cols else c
                    for c in joined_sample_cols
                }

                if (
                    self.source_alias
                    and self.target_alias
                    and self.source != self.source_alias
                    and self.target != self.target_alias
                ):
                    joined_renamed_cols = {
                        c: (
                            n.replace(
                                "s__",
                                f"{self.source_alias.upper()}__",
                            )
                            if n.startswith("s__")
                            else n
                        )
                        for c, n in joined_renamed_cols.items()
                    }
                    joined_renamed_cols = {
                        c: (
                            n.replace(
                                "t__",
                                f"{self.target_alias.upper()}__",
                            )
                            if n.startswith("t__")
                            else n
                        )
                        for c, n in joined_renamed_cols.items()
                    }

                joined_sample = sample[sample[SQLMESH_SAMPLE_TYPE_COL] == "common_rows"][
                    joined_sample_cols
                ]
                joined_sample.rename(
                    columns=joined_renamed_cols,
                    inplace=True,
                )

                s_sample = sample[sample[SQLMESH_SAMPLE_TYPE_COL] == "source_only"][
                    [
                        *[f"s__{c}" for c in s_index_names],
                        *[f"s__{c}" for c in source_schema if c not in s_index_names],
                    ]
                ]
                s_sample.rename(
                    columns={c: c.replace("s__", "") for c in s_sample.columns}, inplace=True
                )

                t_sample = sample[sample[SQLMESH_SAMPLE_TYPE_COL] == "target_only"][
                    [
                        *[f"t__{c}" for c in t_index_names],
                        *[f"t__{c}" for c in target_schema if c not in t_index_names],
                    ]
                ]
                t_sample.rename(
                    columns={c: c.replace("t__", "") for c in t_sample.columns}, inplace=True
                )

                sample.drop(
                    columns=[
                        f"s__{SQLMESH_JOIN_KEY_COL}",
                        f"t__{SQLMESH_JOIN_KEY_COL}",
                        SQLMESH_SAMPLE_TYPE_COL,
                    ],
                    inplace=True,
                )

                self._row_diff = RowDiff(
                    source=self.source,
                    target=self.target,
                    stats=stats,
                    column_stats=column_stats,
                    sample=sample,
                    joined_sample=joined_sample,
                    s_sample=s_sample,
                    t_sample=t_sample,
                    source_alias=self.source_alias,
                    target_alias=self.target_alias,
                    model_name=self.model_name,
                    decimals=self.decimals,
                )

        return self._row_diff

    def _fetch_sample(
        self,
        sample_table: exp.Table,
        s_selects: t.Dict[str, exp.Expr],
        s_index: t.List[exp.Column],
        t_selects: t.Dict[str, exp.Expr],
        t_index: t.List[exp.Column],
        limit: int,
    ) -> pd.DataFrame:
        rendered_data_column_names = [
            name(s) for s in list(s_selects.values()) + list(t_selects.values())
        ]
        sample_type = exp.to_identifier(SQLMESH_SAMPLE_TYPE_COL)

        source_only_sample = (
            exp.select(
                exp.Literal.string("source_only").as_(sample_type), *rendered_data_column_names
            )
            .from_(sample_table)
            .where(exp.and_(exp.column("s_exists").eq(1), exp.column("row_joined").eq(0)))
            .order_by(*(name(s_selects[c.name]) for c in s_index))
            .limit(limit)
        )

        target_only_sample = (
            exp.select(
                exp.Literal.string("target_only").as_(sample_type), *rendered_data_column_names
            )
            .from_(sample_table)
            .where(exp.and_(exp.column("t_exists").eq(1), exp.column("row_joined").eq(0)))
            .order_by(*(name(t_selects[c.name]) for c in t_index))
            .limit(limit)
        )

        common_rows_sample = (
            exp.select(
                exp.Literal.string("common_rows").as_(sample_type), *rendered_data_column_names
            )
            .from_(sample_table)
            .where(exp.and_(exp.column("row_joined").eq(1), exp.column("row_full_match").eq(0)))
            .order_by(
                *(name(s_selects[c.name]) for c in s_index),
                *(name(t_selects[c.name]) for c in t_index),
            )
            .limit(limit)
        )

        query = (
            exp.Select()
            .with_("source_only", source_only_sample)
            .with_("target_only", target_only_sample)
            .with_("common_rows", common_rows_sample)
            .select(sample_type, *rendered_data_column_names)
            .from_("source_only")
            .union(
                exp.select(sample_type, *rendered_data_column_names).from_("target_only"),
                distinct=False,
            )
            .union(
                exp.select(sample_type, *rendered_data_column_names).from_("common_rows"),
                distinct=False,
            )
        )

        return self.adapter.fetchdf(query, quote_identifiers=True)

    # ------------------------------------------------------------------
    # Minimum-change (duplicate-aware) row diff
    # ------------------------------------------------------------------

    def _minimum_change_staged_query(
        self,
        table: exp.Table,
        table_alias: str,
        key_names: t.List[str],
        schema: t.Dict[str, exp.DataType],
        comparable: t.Dict[str, exp.DataType],
    ) -> exp.Query:
        """Build the filtered, key- and signature-annotated staging query for one side."""
        key_expr = self._minimum_change_key_expression(key_names, table_alias)

        # The signature covers every comparable column so equal signatures mean fully-equal rows.
        signature_columns = {c: comparable[c] for c in schema if c in comparable}
        signature_expr = self.adapter.row_signature(
            signature_columns,
            table_alias,
            decimal_precision=self.decimals,
        )

        query = exp.select(
            *(exp.column(c, table_alias) for c in schema),
            key_expr.as_(SQLMESH_JOIN_KEY_COL),
            signature_expr.as_(SQLMESH_SIGNATURE_COL),
        ).from_(table.as_(table_alias))
        if self.where:
            query = query.where(self.where.copy())
        # Two deterministic ordinals are assigned per business key, both independent of physical
        # storage order because of the signature ordering:
        #   * a global row number unique within the key, used for stable display and tie-breaks;
        #   * a per-(key, signature) ordinal. Full matches always consume the lowest ordinals of
        #     every equivalence class, so a residual multiplicity ``k`` is precisely the rows
        #     whose class ordinal is <= k. This expands multiplicities without duplication and
        #     without any cross-side cartesian product.
        query = exp.select(
            exp.Star(),
            exp.Window(
                this=exp.RowNumber(),
                partition_by=[exp.column(SQLMESH_JOIN_KEY_COL)],
                order=exp.Order(expressions=[exp.Ordered(this=exp.column(SQLMESH_SIGNATURE_COL))]),
            ).as_(SQLMESH_ROW_NUMBER_COL),
            exp.Window(
                this=exp.RowNumber(),
                partition_by=[
                    exp.column(SQLMESH_JOIN_KEY_COL),
                    exp.column(SQLMESH_SIGNATURE_COL),
                ],
                order=exp.Order(expressions=[exp.Ordered(this=exp.column(SQLMESH_SIGNATURE_COL))]),
            ).as_(SQLMESH_CLASS_ORDINAL_COL),
        ).from_(query.subquery(table_alias))
        return query

    def _minimum_change_row_diff(self, temp_schema: t.Optional[str] = None) -> RowDiff:
        key_names = self._minimum_change_key_columns()
        source_schema, target_schema, comparable = self._validate_minimum_change_inputs(key_names)
        non_key_comparable = [c for c in comparable if c not in key_names]

        if not temp_schema:
            temp_schema = "sqlmesh_temp"

        from sqlmesh.core.engine_adapter.base import random_id

        base_schema = to_schema(temp_schema, dialect=self.dialect)
        run_id = f"mc_{random_id(short=True)}"
        # DuckDB (and several other engines) do not support schemas nested inside schemas, so
        # each run isolates its staging tables by a unique prefix inside the configured temp
        # schema. Every table is dropped individually in ``finally``, which also guarantees that
        # the caller's temp schema is never removed.
        self.adapter.create_schema(base_schema, ignore_if_exists=True)

        def _table(name: str) -> exp.Table:
            return exp.table_(
                f"{run_id}_{name}",
                db=base_schema.db,
                catalog=base_schema.catalog,
                quoted=True,
            )

        staged_s = _table("staged_s")
        staged_t = _table("staged_t")
        groups_s = _table("groups_s")
        groups_t = _table("groups_t")
        full_match_table = _table("full_matches")
        capped_s = _table("residual_s")
        capped_t = _table("residual_t")
        key_residual_s = _table("key_res_s")
        key_residual_t = _table("key_res_t")
        created: t.List[exp.Table] = []

        try:
            # 1) Stage the filtered rows with the collision-free key, row signature and a stable
            #    per-key ordinal. All heavy scanning happens inside the database.
            self.adapter.ctas(
                staged_s,
                self._minimum_change_staged_query(
                    self.source_table, "s", key_names, source_schema, comparable
                ),
                exists=False,
            )
            created.append(staged_s)
            self.adapter.ctas(
                staged_t,
                self._minimum_change_staged_query(
                    self.target_table, "t", key_names, target_schema, comparable
                ),
                exists=False,
            )
            created.append(staged_t)

            # 2) Equivalence classes within each key: rows grouped by (key, signature).
            group_cols = [SQLMESH_JOIN_KEY_COL, SQLMESH_SIGNATURE_COL]
            self.adapter.ctas(
                groups_s,
                exp.select(
                    exp.column(SQLMESH_JOIN_KEY_COL),
                    exp.column(SQLMESH_SIGNATURE_COL),
                    exp.func("COUNT", exp.Star()).as_(SQLMESH_MULTIPLICITY_COL),
                )
                .from_(staged_s)
                .group_by(*group_cols),
                exists=False,
            )
            created.append(groups_s)
            self.adapter.ctas(
                groups_t,
                exp.select(
                    exp.column(SQLMESH_JOIN_KEY_COL),
                    exp.column(SQLMESH_SIGNATURE_COL),
                    exp.func("COUNT", exp.Star()).as_(SQLMESH_MULTIPLICITY_COL),
                )
                .from_(staged_t)
                .group_by(*group_cols),
                exists=False,
            )
            created.append(groups_t)

            # 3) Real row counts before any pairing.
            s_count = int(
                self.adapter.fetchone(exp.select(exp.func("COUNT", exp.Star())).from_(staged_s))[0]
            )
            t_count = int(
                self.adapter.fetchone(exp.select(exp.func("COUNT", exp.Star())).from_(staged_t))[0]
            )
            if s_count == 0 and t_count == 0:
                return self._empty_minimum_change_row_diff(non_key_comparable)

            # 4) Full matches: pair identical (key, signature) classes at min multiplicity.
            full_match_query = (
                exp.select(
                    exp.func(
                        "COALESCE",
                        exp.column(SQLMESH_JOIN_KEY_COL, "gs"),
                        exp.column(SQLMESH_JOIN_KEY_COL, "gt"),
                    ).as_(SQLMESH_JOIN_KEY_COL),
                    exp.func(
                        "COALESCE",
                        exp.column(SQLMESH_SIGNATURE_COL, "gs"),
                        exp.column(SQLMESH_SIGNATURE_COL, "gt"),
                    ).as_(SQLMESH_SIGNATURE_COL),
                    exp.func(
                        "LEAST",
                        exp.func("COALESCE", exp.column(SQLMESH_MULTIPLICITY_COL, "gs"), 0),
                        exp.func("COALESCE", exp.column(SQLMESH_MULTIPLICITY_COL, "gt"), 0),
                    ).as_("full_cnt"),
                    exp.func("COALESCE", exp.column(SQLMESH_MULTIPLICITY_COL, "gs"), 0).as_(
                        "s_cnt"
                    ),
                    exp.func("COALESCE", exp.column(SQLMESH_MULTIPLICITY_COL, "gt"), 0).as_(
                        "t_cnt"
                    ),
                )
                .from_(groups_s.as_("gs"))
                .join(
                    groups_t.as_("gt"),
                    on=exp.and_(
                        exp.NullSafeEQ(
                            this=exp.column(SQLMESH_JOIN_KEY_COL, "gs"),
                            expression=exp.column(SQLMESH_JOIN_KEY_COL, "gt"),
                        ),
                        exp.NullSafeEQ(
                            this=exp.column(SQLMESH_SIGNATURE_COL, "gs"),
                            expression=exp.column(SQLMESH_SIGNATURE_COL, "gt"),
                        ),
                    ),
                    join_type="FULL",
                )
            )
            self.adapter.ctas(full_match_table, full_match_query, exists=False)
            created.append(full_match_table)
            full_match_count = int(
                self.adapter.fetchone(
                    exp.select(
                        exp.func("COALESCE", exp.func("SUM", exp.column("full_cnt")), 0)
                    ).from_(full_match_table)
                )[0]
            )

            # 5) Residual multiplicities per (key, signature) after consuming full matches. The
            #    full matches always take the lowest class ordinals, so a residual copy is any
            #    staged row whose class ordinal is strictly greater than ``full_cnt``.
            self.adapter.ctas(
                capped_s,
                exp.select(
                    exp.column(SQLMESH_JOIN_KEY_COL),
                    exp.column(SQLMESH_SIGNATURE_COL),
                    exp.column("full_cnt"),
                    (exp.column("s_cnt") - exp.column("full_cnt")).as_("residual_cnt"),
                )
                .from_(full_match_table)
                .where(exp.column("s_cnt") - exp.column("full_cnt") > 0),
                exists=False,
            )
            created.append(capped_s)
            self.adapter.ctas(
                capped_t,
                exp.select(
                    exp.column(SQLMESH_JOIN_KEY_COL),
                    exp.column(SQLMESH_SIGNATURE_COL),
                    exp.column("full_cnt"),
                    (exp.column("t_cnt") - exp.column("full_cnt")).as_("residual_cnt"),
                )
                .from_(full_match_table)
                .where(exp.column("t_cnt") - exp.column("full_cnt") > 0),
                exists=False,
            )
            created.append(capped_t)

            # 6) Per-key residual totals; abort if a two-sided key exceeds the cap on either side.
            self.adapter.ctas(
                key_residual_s,
                exp.select(
                    exp.column(SQLMESH_JOIN_KEY_COL),
                    exp.func("SUM", exp.column("residual_cnt")).as_("residual_total"),
                )
                .from_(capped_s)
                .group_by(SQLMESH_JOIN_KEY_COL),
                exists=False,
            )
            created.append(key_residual_s)
            self.adapter.ctas(
                key_residual_t,
                exp.select(
                    exp.column(SQLMESH_JOIN_KEY_COL),
                    exp.func("SUM", exp.column("residual_cnt")).as_("residual_total"),
                )
                .from_(capped_t)
                .group_by(SQLMESH_JOIN_KEY_COL),
                exists=False,
            )
            created.append(key_residual_t)

            over_cap_rows = self.adapter.fetchdf(
                exp.select(
                    exp.func(
                        "COALESCE",
                        exp.column(SQLMESH_JOIN_KEY_COL, "ks"),
                        exp.column(SQLMESH_JOIN_KEY_COL, "kt"),
                    ).as_(SQLMESH_JOIN_KEY_COL),
                    exp.func("COALESCE", exp.column("residual_total", "ks"), 0).as_("s_residual"),
                    exp.func("COALESCE", exp.column("residual_total", "kt"), 0).as_("t_residual"),
                )
                .from_(key_residual_s.as_("ks"))
                .join(
                    key_residual_t.as_("kt"),
                    on=exp.NullSafeEQ(
                        this=exp.column(SQLMESH_JOIN_KEY_COL, "ks"),
                        expression=exp.column(SQLMESH_JOIN_KEY_COL, "kt"),
                    ),
                    join_type="FULL",
                )
                .where(
                    # Only two-sided keys can require an exact pairing; a key whose residuals
                    # exist on just one side is reported by multiplicity without any solving.
                    exp.and_(
                        exp.column("s_residual") > 0,
                        exp.column("t_residual") > 0,
                        exp.or_(
                            exp.column("s_residual") > self.max_unmatched_rows,
                            exp.column("t_residual") > self.max_unmatched_rows,
                        ),
                    )
                )
                .order_by(SQLMESH_JOIN_KEY_COL)
            )
            if not over_cap_rows.empty:
                details = "; ".join(
                    f"key {row[SQLMESH_JOIN_KEY_COL]!r}: "
                    f"{int(row['s_residual'])} source vs "
                    f"{int(row['t_residual'])} target unmatched copies"
                    for _, row in over_cap_rows.iterrows()
                )
                raise SQLMeshError(
                    "row_match='minimum_change' cannot produce an exact result: one or more "
                    f"business keys have more than max_unmatched_rows={self.max_unmatched_rows} "
                    "unmatched copies on a side while the other side is non-empty. "
                    f"{details}. Raise max_unmatched_rows to attempt the comparison."
                )

            return self._solve_minimum_change(
                staged_s=staged_s,
                staged_t=staged_t,
                capped_s=capped_s,
                capped_t=capped_t,
                key_residual_s=key_residual_s,
                key_residual_t=key_residual_t,
                source_schema=source_schema,
                target_schema=target_schema,
                key_names=key_names,
                non_key_comparable=non_key_comparable,
                s_count=s_count,
                t_count=t_count,
                full_match_count=full_match_count,
            )
        finally:
            for table in reversed(created):
                self.adapter.drop_table(table, exists=True)

    def _empty_minimum_change_row_diff(self, non_key_comparable: t.List[str]) -> RowDiff:
        import pandas as pd

        stats = {
            "s_count": 0,
            "t_count": 0,
            "join_count": 0,
            "full_match_count": 0,
            "partial_match_count": 0,
            "s_only_count": 0,
            "t_only_count": 0,
            "changed_cell_count": 0,
            "null_grain_count": 0,
            "distinct_count_s": 0,
            "distinct_count_t": 0,
        }
        column_stats = pd.DataFrame(
            {"pct_match": [], "changed_cells": []},
            index=pd.Index([], name="column"),
        )
        empty = pd.DataFrame()
        return RowDiff(
            source=self.source,
            target=self.target,
            stats=stats,
            column_stats=column_stats,
            sample=empty,
            joined_sample=empty,
            s_sample=empty,
            t_sample=empty,
            source_alias=self.source_alias,
            target_alias=self.target_alias,
            model_name=self.model_name,
            decimals=self.decimals,
            row_match=ROW_MATCH_MINIMUM_CHANGE,
        )

    def _residual_row_query(
        self,
        capped_table: exp.Table,
        staged_table: exp.Table,
        schema: t.Dict[str, exp.DataType],
        key_value: t.Any,
    ) -> exp.Query:
        """Individual residual rows of one key, bounded by the cap, without any cross-side join.

        Residual (key, signature) classes carry how many of their copies remain after the full
        matches were consumed. Because every staged row has a unique per-key ordinal (see
        :meth:`_minimum_change_staged_query`) and equal rows are ordered contiguously by
        signature, selecting the rows whose ordinal does not exceed that multiplicity returns
        exactly the residual copies, with no duplication and no cross-side cartesian product.
        """
        key_column = exp.column(SQLMESH_JOIN_KEY_COL, "cap")
        key_predicate: exp.Expr
        if key_value is None:
            key_predicate = key_column.is_(exp.Null())
        else:
            # The join key is either the encoded composite string or a native scalar; render the
            # Python value as a typed literal rather than assuming a string.
            key_predicate = key_column.eq(exp.convert(key_value))

        return (
            exp.select(*(exp.column(c, "st") for c in schema))
            .from_(capped_table.as_("cap"))
            .join(
                staged_table.as_("st"),
                on=exp.and_(
                    exp.NullSafeEQ(
                        this=exp.column(SQLMESH_JOIN_KEY_COL, "cap"),
                        expression=exp.column(SQLMESH_JOIN_KEY_COL, "st"),
                    ),
                    exp.NullSafeEQ(
                        this=exp.column(SQLMESH_SIGNATURE_COL, "cap"),
                        expression=exp.column(SQLMESH_SIGNATURE_COL, "st"),
                    ),
                ),
            )
            .where(
                key_predicate,
                # full matches consumed the lowest class ordinals; residuals are the remainder
                exp.column(SQLMESH_CLASS_ORDINAL_COL, "st") > exp.column("full_cnt", "cap"),
            )
            .order_by(exp.column(SQLMESH_ROW_NUMBER_COL, "st"))
        )

    def _solve_minimum_change(
        self,
        *,
        staged_s: exp.Table,
        staged_t: exp.Table,
        capped_s: exp.Table,
        capped_t: exp.Table,
        key_residual_s: exp.Table,
        key_residual_t: exp.Table,
        source_schema: t.Dict[str, exp.DataType],
        target_schema: t.Dict[str, exp.DataType],
        key_names: t.List[str],
        non_key_comparable: t.List[str],
        s_count: int,
        t_count: int,
        full_match_count: int,
    ) -> RowDiff:
        import pandas as pd

        both_side_keys = [
            row[0]
            for row in self.adapter.fetchall(
                exp.select(exp.column(SQLMESH_JOIN_KEY_COL, "ks").as_(SQLMESH_JOIN_KEY_COL))
                .from_(key_residual_s.as_("ks"))
                .join(
                    key_residual_t.as_("kt"),
                    on=exp.NullSafeEQ(
                        this=exp.column(SQLMESH_JOIN_KEY_COL, "ks"),
                        expression=exp.column(SQLMESH_JOIN_KEY_COL, "kt"),
                    ),
                    join_type="INNER",
                )
                .order_by(exp.column(SQLMESH_JOIN_KEY_COL, "ks"))
            )
        ]

        changed_pair_count = 0
        changed_cell_count = 0
        column_changed_cells = {c: 0 for c in non_key_comparable}
        sample_rows: t.List[dict] = []
        s_only_count = 0
        t_only_count = 0

        for key_value in both_side_keys:
            s_rows = self.adapter.fetchdf(
                self._residual_row_query(capped_s, staged_s, source_schema, key_value),
                quote_identifiers=True,
            ).to_dict("records")
            t_rows = self.adapter.fetchdf(
                self._residual_row_query(capped_t, staged_t, target_schema, key_value),
                quote_identifiers=True,
            ).to_dict("records")

            cost_matrix: t.List[t.List[int]] = []
            match_flags: t.List[t.List[t.List[int]]] = []
            for s_r in s_rows:
                row_costs: t.List[int] = []
                row_flags: t.List[t.List[int]] = []
                for t_r in t_rows:
                    flags = [0 if self._cells_equal(s_r, t_r, c) else 1 for c in non_key_comparable]
                    row_flags.append(flags)
                    row_costs.append(sum(flags))
                cost_matrix.append(row_costs)
                match_flags.append(row_flags)

            pairs, _ = minimum_change_assignment(cost_matrix)
            paired_s = {i for i, _ in pairs}
            paired_t = {j for _, j in pairs}

            for i, j in pairs:
                changed_pair_count += 1
                diffs = [c for c, flag in zip(non_key_comparable, match_flags[i][j]) if flag == 1]
                changed_cell_count += len(diffs)
                for c, flag in zip(non_key_comparable, match_flags[i][j]):
                    column_changed_cells[c] += flag
                if len(sample_rows) < self.limit:
                    sample_rows.append(
                        self._pair_sample_row(
                            s_rows[i], t_rows[j], key_names, non_key_comparable, diffs
                        )
                    )

            for idx, s_r in enumerate(s_rows):
                if idx not in paired_s:
                    s_only_count += 1
                    if len(sample_rows) < self.limit:
                        sample_rows.append(
                            self._side_only_sample_row(s_r, key_names, source_schema, "s")
                        )
            for idx, t_r in enumerate(t_rows):
                if idx not in paired_t:
                    t_only_count += 1
                    if len(sample_rows) < self.limit:
                        sample_rows.append(
                            self._side_only_sample_row(t_r, key_names, target_schema, "t")
                        )

        # Single-side residual groups need no pairing: every copy is reported as side-only.
        s_only_count += self._count_single_side_residuals(key_residual_s, key_residual_t)
        t_only_count += self._count_single_side_residuals(key_residual_t, key_residual_s)

        sample_rows.extend(
            self._sample_single_side_rows(
                key_residual_s,
                key_residual_t,
                capped_s,
                staged_s,
                key_names,
                source_schema,
                "s",
                max(0, self.limit - len(sample_rows)),
            )
        )
        sample_rows.extend(
            self._sample_single_side_rows(
                key_residual_t,
                key_residual_s,
                capped_t,
                staged_t,
                key_names,
                target_schema,
                "t",
                max(0, self.limit - len(sample_rows)),
            )
        )

        # Conservation invariants against the real row counts.
        assert s_count == full_match_count + changed_pair_count + s_only_count
        assert t_count == full_match_count + changed_pair_count + t_only_count

        join_count = full_match_count + changed_pair_count
        distinct_s = int(
            self.adapter.fetchone(
                exp.select(
                    exp.func(
                        "COUNT",
                        exp.Distinct(expressions=[exp.column(SQLMESH_JOIN_KEY_COL)]),
                    )
                ).from_(staged_s)
            )[0]
        )
        distinct_t = int(
            self.adapter.fetchone(
                exp.select(
                    exp.func(
                        "COUNT",
                        exp.Distinct(expressions=[exp.column(SQLMESH_JOIN_KEY_COL)]),
                    )
                ).from_(staged_t)
            )[0]
        )
        null_grain = int(
            self.adapter.fetchone(
                exp.select(exp.func("COUNT", exp.Star()))
                .from_(staged_s)
                .where(exp.column(SQLMESH_JOIN_KEY_COL).is_(exp.Null()))
            )[0]
        )

        stats = {
            "s_count": s_count,
            "t_count": t_count,
            "join_count": join_count,
            "full_match_count": full_match_count,
            "partial_match_count": changed_pair_count,
            "s_only_count": s_only_count,
            "t_only_count": t_only_count,
            "changed_cell_count": changed_cell_count,
            "null_grain_count": null_grain,
            "distinct_count_s": distinct_s,
            "distinct_count_t": distinct_t,
        }

        # Column stats over all comparable pairs (full matches + changed pairings).
        total_pairs = join_count
        column_index = pd.Index(non_key_comparable, name="column")
        pct_values = [
            round(100 * (total_pairs - column_changed_cells[c]) / total_pairs, 6)
            if total_pairs
            else 100.0
            for c in non_key_comparable
        ]
        column_stats = pd.DataFrame(
            {
                "pct_match": pct_values,
                "changed_cells": [column_changed_cells[c] for c in non_key_comparable],
            },
            index=column_index,
        )

        sample_df = pd.DataFrame(sample_rows)
        joined_sample, s_sample, t_sample = self._split_minimum_change_samples(
            sample_df, source_schema, target_schema, key_names
        )

        return RowDiff(
            source=self.source,
            target=self.target,
            stats=stats,
            column_stats=column_stats,
            sample=sample_df,
            joined_sample=joined_sample,
            s_sample=s_sample,
            t_sample=t_sample,
            source_alias=self.source_alias,
            target_alias=self.target_alias,
            model_name=self.model_name,
            decimals=self.decimals,
            row_match=ROW_MATCH_MINIMUM_CHANGE,
        )

    def _count_single_side_residuals(self, this_side: exp.Table, other_side: exp.Table) -> int:
        row = self.adapter.fetchone(
            exp.select(
                exp.func("COALESCE", exp.func("SUM", exp.column("residual_total", "this")), 0)
            )
            .from_(this_side.as_("this"))
            .join(
                other_side.as_("other"),
                on=exp.NullSafeEQ(
                    this=exp.column(SQLMESH_JOIN_KEY_COL, "this"),
                    expression=exp.column(SQLMESH_JOIN_KEY_COL, "other"),
                ),
                join_type="LEFT",
            )
            .where(exp.column(SQLMESH_JOIN_KEY_COL, "other").is_(exp.Null()))
        )
        return int(row[0])

    def _sample_single_side_rows(
        self,
        this_side: exp.Table,
        other_side: exp.Table,
        capped_table: exp.Table,
        staged_table: exp.Table,
        key_names: t.List[str],
        schema: t.Dict[str, exp.DataType],
        side: str,
        budget: int,
    ) -> t.List[dict]:
        if budget <= 0:
            return []
        query = (
            exp.select(*(exp.column(c, "st") for c in schema))
            .from_(this_side.as_("this"))
            .join(
                other_side.as_("other"),
                on=exp.NullSafeEQ(
                    this=exp.column(SQLMESH_JOIN_KEY_COL, "this"),
                    expression=exp.column(SQLMESH_JOIN_KEY_COL, "other"),
                ),
                join_type="LEFT",
            )
            .join(
                capped_table.as_("cap"),
                on=exp.NullSafeEQ(
                    this=exp.column(SQLMESH_JOIN_KEY_COL, "this"),
                    expression=exp.column(SQLMESH_JOIN_KEY_COL, "cap"),
                ),
            )
            .join(
                staged_table.as_("st"),
                on=exp.and_(
                    exp.NullSafeEQ(
                        this=exp.column(SQLMESH_JOIN_KEY_COL, "cap"),
                        expression=exp.column(SQLMESH_JOIN_KEY_COL, "st"),
                    ),
                    exp.NullSafeEQ(
                        this=exp.column(SQLMESH_SIGNATURE_COL, "cap"),
                        expression=exp.column(SQLMESH_SIGNATURE_COL, "st"),
                    ),
                ),
            )
            .where(
                exp.column(SQLMESH_JOIN_KEY_COL, "other").is_(exp.Null()),
                exp.column(SQLMESH_CLASS_ORDINAL_COL, "st") > exp.column("full_cnt", "cap"),
            )
            .order_by(
                exp.column(SQLMESH_JOIN_KEY_COL, "st"),
                exp.column(SQLMESH_ROW_NUMBER_COL, "st"),
            )
            .limit(budget)
        )
        df = self.adapter.fetchdf(query, quote_identifiers=True)
        return [
            self._side_only_sample_row(record, key_names, schema, side)
            for record in df.to_dict("records")
        ]

    @staticmethod
    def _cells_equal(s_row: dict, t_row: dict, column: str) -> bool:
        import math

        import pandas as pd

        def _is_null(value: t.Any) -> bool:
            if value is None:
                return True
            try:
                if math.isnan(value):
                    return True
            except (TypeError, ValueError):
                pass
            try:
                return bool(pd.isna(value))
            except (TypeError, ValueError):
                return False

        s_val, t_val = s_row[column], t_row[column]
        s_null, t_null = _is_null(s_val), _is_null(t_val)
        if s_null or t_null:
            return s_null and t_null
        return s_val == t_val

    def _pair_sample_row(
        self,
        s_row: dict,
        t_row: dict,
        key_names: t.List[str],
        non_key_comparable: t.List[str],
        diffs: t.List[str],
    ) -> dict:
        row: t.Dict[str, t.Any] = {SQLMESH_SAMPLE_TYPE_COL: "common_rows"}
        for k in key_names:
            row[k] = s_row[k]
        for c in non_key_comparable:
            row[f"s__{c}"] = s_row[c]
            row[f"t__{c}"] = t_row[c]
        row[SQLMESH_MULTIPLICITY_COL] = 1
        row["changed_columns"] = ",".join(diffs)
        return row

    def _side_only_sample_row(
        self, record: dict, key_names: t.List[str], schema: t.Dict[str, exp.DataType], side: str
    ) -> dict:
        prefix = "s" if side == "s" else "t"
        row: t.Dict[str, t.Any] = {
            SQLMESH_SAMPLE_TYPE_COL: "source_only" if side == "s" else "target_only"
        }
        for k in key_names:
            row[k] = record[k]
        # Business keys are already emitted unprefixed above; only prefix the non-key columns.
        for c in schema:
            if c not in key_names:
                row[f"{prefix}__{c}"] = record[c]
        row[SQLMESH_MULTIPLICITY_COL] = 1
        row["changed_columns"] = ""
        return row

    def _split_minimum_change_samples(
        self,
        sample_df: "pd.DataFrame",
        source_schema: t.Dict[str, exp.DataType],
        target_schema: t.Dict[str, exp.DataType],
        key_names: t.List[str],
    ) -> t.Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        import pandas as pd

        if sample_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        joined = sample_df[sample_df[SQLMESH_SAMPLE_TYPE_COL] == "common_rows"].copy()
        s_only = sample_df[sample_df[SQLMESH_SAMPLE_TYPE_COL] == "source_only"].copy()
        t_only = sample_df[sample_df[SQLMESH_SAMPLE_TYPE_COL] == "target_only"].copy()
        # The sample-type marker and the pair's changed-column list are internal control columns;
        # the multiplicity is retained because the console uses it to weight merged sample rows.
        for frame in (joined, s_only, t_only):
            frame.drop(
                columns=[SQLMESH_SAMPLE_TYPE_COL, "changed_columns"],
                errors="ignore",
                inplace=True,
            )
        return joined, s_only, t_only


def name(e: exp.Expr) -> str:
    return e.args["alias"].sql(identify=True)
