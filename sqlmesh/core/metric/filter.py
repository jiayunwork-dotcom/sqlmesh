"""Metric ``filter`` support: parsing, validation, decomposition and rewriting.

A metric filter is a SQL boolean expression whose columns are fully qualified
with model names, e.g. ``demo.tags.label = 'trial'``.  Filters are only allowed
on base (aggregation) metrics and are used exclusively to decide whether a
fact row is *eligible*; they never introduce new grouping dimensions and the
measure is never multiplied by the number of matching related rows.

The expressions accepted by the first version are intentionally small:

* column vs constant comparisons (``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``),
* ``IS NULL`` / ``IS NOT NULL``,
* boolean combinations with ``AND``, ``OR``, ``NOT`` and parentheses.

A filter is decomposed into one predicate per *region* of related models.  A
region is a maximal subtree whose columns all belong to the same model.  Each
region is evaluated against the rows of that model independently (same-row
semantics: all conditions of a region must be satisfied by one record), while
the boolean operators that connect regions are evaluated with SQL three valued
logic where a missing related object behaves like a row of all ``NULL`` values.
"""

from __future__ import annotations

import typing as t

from sqlglot import exp

from sqlmesh.utils.errors import ConfigError

if t.TYPE_CHECKING:
    from sqlmesh.core.reference import ReferenceGraph, ReferencePath

# Comparison operators that may appear in filters.
_COMPARISONS = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
)


def _remove_namespace(table: str) -> str:
    """Rewrite catalog/schema namespaces like catalog.db to catalog__db."""
    return table.replace('"', "").replace(".", "__")


def column_model(column: exp.Column) -> str:
    """The fully qualified model name of a fully qualified filter column."""
    return ".".join(part.sql() for part in column.parts[:-1])


def _strip_catalog(name: str) -> str:
    """Drop the leading catalog segment, if any."""
    parts = name.split(".")
    return ".".join(parts[1:]) if len(parts) > 2 else name

# Boolean node types that may combine predicates.
_BOOLEANS = (exp.And, exp.Or, exp.Not, exp.Paren)

# Leaf predicates (besides boolean combinations).
_LEAVES = (*_COMPARISONS, exp.Is)


class Filter:
    """A validated metric filter attached to a base metric.

    Args:
        expression: The original, fully qualified boolean expression.
        dialect: The dialect the expression was written in.
        metric_name: The name of the metric owning the filter, for errors.
        path: The source path of the metric definition, for errors.
    """

    def __init__(
        self,
        expression: exp.Expr,
        dialect: str = "",
        metric_name: str = "",
        path: t.Any = "",
    ):
        self.expression = expression
        self.dialect = dialect
        self.metric_name = metric_name
        self.path = path
        # Models referenced by the filter, in fully qualified form.
        self.models: t.Set[str] = set()
        self._validate(expression)

    def _error(self, msg: str) -> ConfigError:
        where = f" in filter of metric '{self.metric_name}'" if self.metric_name else " in filter"
        return ConfigError(f"{msg}{where}. '{self.path}'")

    def _validate(self, node: exp.Expr) -> None:
        if isinstance(node, exp.Column):
            if not node.table:
                raise self._error(
                    f"Filter column '{node.sql()}' must be fully qualified with its model name"
                )
            self.models.add(column_model(node))
            return

        # Reject actual scalar functions (LOWER, COALESCE, ...) but not the
        # boolean operators AND/OR, which also inherit from Func in sqlglot.
        if isinstance(node, exp.Func) and not isinstance(node, _BOOLEANS):
            raise self._error(
                f"Unsupported filter expression '{node.sql()}'. Function calls are not "
                "supported; only column/constant comparisons, IS NULL and boolean "
                "combinations (AND, OR, NOT, parentheses) are allowed"
            )

        if isinstance(node, _LEAVES):
            self._validate_leaf(node)
            return

        if isinstance(node, _BOOLEANS):
            for child in iter_child_expressions(node):
                self._validate(child)
            return

        raise self._error(
            f"Unsupported filter expression '{node.sql()}'. Only column/constant comparisons, "
            "IS NULL and boolean combinations (AND, OR, NOT, parentheses) are supported"
        )

    def _validate_leaf(self, node: exp.Expr) -> None:
        # A scalar function anywhere inside a leaf predicate is unsupported.
        for child in node.walk():
            if isinstance(child, exp.Func) and not isinstance(child, _BOOLEANS):
                raise self._error(
                    f"Unsupported filter expression '{node.sql()}'. Function calls are not "
                    "supported; only column/constant comparisons, IS NULL and boolean "
                    "combinations (AND, OR, NOT, parentheses) are allowed"
                )

        columns = list(node.find_all(exp.Column))

        if not columns:
            raise self._error(f"Filter predicate '{node.sql()}' does not reference any column")

        if isinstance(node, exp.Is):
            # IS NULL / IS NOT NULL - the right hand side is NULL / TRUE / FALSE.
            if not isinstance(node.expression, (exp.Null, exp.Boolean)):
                raise self._error(
                    f"Unsupported IS expression '{node.sql()}'. Only IS NULL / IS NOT NULL are supported"
                )
        else:
            models = {column_model(column) for column in columns}
            if len(models) != 1:
                raise self._error(
                    f"Filter predicate '{node.sql()}' must reference a single model, "
                    "combine cross-model predicates with AND/OR instead"
                )
            non_column_nodes = [
                child
                for child in iter_child_expressions(node)
                if not list(child.find_all(exp.Column))
            ]
            if len(columns) != 1 or len(non_column_nodes) != 1:
                raise self._error(
                    f"Filter comparison '{node.sql()}' must compare one column with one constant"
                )
            _validate_constant(non_column_nodes[0], self._error)

        # Recurse into the columns so the model set is fully populated.
        for column in columns:
            self._validate(column)
    @property
    def regions(self) -> t.List[FilterRegion]:
        """The maximal single-model sub-expressions (regions) of this filter.

        Each region is evaluated against the rows of one related model.  The
        remaining tree only contains boolean operators with region placeholders
        and predicates on the fact model itself.
        """
        return _extract_regions(self.expression, self)

    def rewrite_with_regions(
        self,
        fact_model: str,
        graph: ReferenceGraph,
        model_aliases: t.Dict[str, str],
        region_flags: t.Dict[t.Tuple[str, str], exp.Column],
    ) -> exp.Expr:
        """Build the fact-level eligibility expression of this filter.

        Every related-model region is replaced by its boolean flag column;
        fact-model predicates stay inline with flat table aliases. Region
        subtrees are matched bottom-up so a grouped same-model region (for
        example an AND built from several operands) is replaced as a whole by
        one flag.
        """
        fact_names = {fact_model, _strip_catalog(fact_model)}

        # Map each related model to the flag of the region built for it. When a
        # model has one region (the common case) every subtree of that model is
        # replaced by the same flag, so an AND of two same-model predicates
        # collapses into one flag reference rather than being stitched back
        # together from separate column predicates.
        model_to_flag: t.Dict[str, exp.Column] = {}
        for region in self.regions:
            if region.model in fact_names:
                continue
            flag = region_flags.get(region.key)
            if flag is not None:
                model_to_flag[region.model] = flag

        def replace_node(node: exp.Expr) -> exp.Expr:
            if isinstance(node, exp.Column):
                alias = model_aliases.get(column_model(node))
                if alias:
                    return exp.column(node.name, table=alias)
                return node.copy()

            models = _models_of(node) if list(node.find_all(exp.Column)) else set()
            if len(models) == 1:
                model = next(iter(models))
                flag = model_to_flag.get(model)
                if flag is not None:
                    return flag.copy()

            return node

        return self.expression.transform(replace_node, copy=True)

        # Rewrite remaining (fact-model) columns to flat aliases.
        for column in expression.find_all(exp.Column):
            alias = model_aliases.get(column_model(column))
            if alias:
                column.set("table", exp.to_identifier(alias))
                column.set("db", None)
                column.set("catalog", None)

        return expression

    def sql(self) -> str:
        return self.expression.sql(dialect=self.dialect)


class FilterRegion:
    """A maximal subtree of a filter whose columns all belong to one model."""

    def __init__(self, expression: exp.Expr, model: str):
        self.expression = expression
        self.model = model

    @property
    def key(self) -> t.Tuple[str, str]:
        # Canonicalize so equivalent formatting (extra parentheses, spacing)
        # shares one flag. Unwrap redundant outer parentheses first.
        expression = self.expression
        while isinstance(expression, exp.Paren):
            expression = expression.unnest()
        return (self.model, " ".join(expression.sql().split()))

    def predicate(self, alias: str) -> exp.Expr:
        """Copy of the region expression with columns qualified by ``alias``."""
        return self.expression.transform(
            lambda node: (
                exp.column(node.name, table=alias) if isinstance(node, exp.Column) else node
            ),
            copy=True,
        )


def iter_child_expressions(node: exp.Expr) -> t.Iterable[exp.Expr]:
    """Iterate the boolean-relevant child expressions of a filter node."""
    if isinstance(node, (exp.Not, exp.Paren)):
        if node.this is not None:
            yield node.this
        return

    if isinstance(node, (exp.And, exp.Or)):
        # sqlglot stores binary predicates in this/expression; chain traversal
        # (a AND b AND c) keeps them reachable without relying on `.expressions`.
        if node.this is not None:
            yield node.this
        if node.expression is not None:
            yield node.expression
        return

    for value in node.args.values():
        if isinstance(value, exp.Expression):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, exp.Expression):
                    yield item


def _validate_constant(node: exp.Expr, error: t.Callable[[str], ConfigError]) -> None:
    if isinstance(node, (exp.Literal, exp.Null, exp.Boolean)):
        return
    if isinstance(node, exp.Paren):
        _validate_constant(node.this, error)
        return
    if isinstance(node, exp.Unary) and isinstance(node.this, exp.Literal):
        # Negative numeric literals parse as Neg(Literal).
        return
    raise error(f"Unsupported filter constant '{node.sql()}'")


def _models_of(node: exp.Expr) -> t.Set[str]:
    return {column_model(column) for column in node.find_all(exp.Column)}


def _flatten(node: exp.Expr, operator: t.Type[exp.Expression]) -> t.List[exp.Expr]:
    """Flatten a left/right-associative chain of ``operator`` into operands."""
    if isinstance(node, exp.Paren):
        return _flatten(node.unnest(), operator)
    if isinstance(node, operator):
        result: t.List[exp.Expr] = []
        for child in iter_child_expressions(node):
            result.extend(_flatten(child, operator))
        return result
    return [node]


def _extract_regions(expression: exp.Expr, flt: Filter) -> t.List[FilterRegion]:
    """Find maximal single-model sub-expressions (regions).

    The filter is split along its top-level boolean operators into operands;
    operands whose columns all belong to one model form one region. Keeping a
    same-model conjunction together is essential for same-row semantics:
    ``activities.kind = 'promo' AND activities.status = 'attended'`` must be
    evaluated by one BOOL_OR over the whole predicate rather than combining the
    two column predicates independently (which would stitch together conditions
    satisfied by different rows of the multi-valued table).
    """
    top_models = _models_of(expression)
    if len(top_models) == 1:
        return [FilterRegion(expression, next(iter(top_models)))]

    # Flatten the whole AND chain, including operands nested on either side,
    # then group consecutive operands by model so same-model predicates form
    # one region regardless of where they sit in the left-associative parse.
    operands = _flatten(expression, exp.And)

    regions: t.List[FilterRegion] = []
    seen: t.Set[t.Tuple[str, str]] = set()
    grouped: t.Dict[str, t.List[exp.Expr]] = {}

    for operand in operands:
        models = _models_of(operand)
        if len(models) != 1:
            # An operand that still mixes models (e.g. an OR subtree) is split
            # recursively and its homogeneous pieces become regions directly.
            for child in iter_child_expressions(operand):
                child_models = _models_of(child)
                if len(child_models) == 1:
                    region = FilterRegion(child, next(iter(child_models)))
                    if region.key not in seen:
                        seen.add(region.key)
                        regions.append(region)
            continue
        grouped.setdefault(next(iter(models)), []).append(operand)

    for model, predicates in grouped.items():
        if len(predicates) == 1:
            region_node = predicates[0]
        else:
            region_node = exp.and_(*predicates, copy=False)
        region = FilterRegion(region_node, model)
        if region.key not in seen:
            seen.add(region.key)
            regions.append(region)

    return regions


def validate_columns(
    flt: Filter,
    graph: ReferenceGraph,
    fact_model: str,
    metric_name: str,
    path: t.Any = "",
) -> t.Dict[str, "ReferencePath"]:
    """Resolve every filter model against the reference graph.

    Returns one :class:`~sqlmesh.core.reference.ReferencePath` per filter
    region, keyed by the region flag name.  Raises a locatable error when a
    column does not exist, no path exists or multiple equally short paths make
    the join ambiguous.
    """
    paths: t.Dict[t.Tuple[str, str], ReferencePath] = {}
    resolved_fact = graph._resolve_model_name(fact_model) or fact_model
    for region in flt.regions:
        model = graph._resolve_model_name(region.model) or region.model

        for column in region.expression.find_all(exp.Column):
            if column.name not in graph.columns_of(model):
                raise ConfigError(
                    f"Column '{column.name}' does not exist on model '{region.model}' referenced by "
                    f"filter of metric '{metric_name}'. '{path}'"
                )

        if model != resolved_fact:
            resolved = graph.find_eligibility_path(resolved_fact, model)
            for ref in resolved:
                for column in ref.columns:
                    if column not in graph.columns_of(ref.model_name):
                        raise ConfigError(
                            f"Reference key '{column}' does not exist on model "
                            f"'{ref.model_name}' while resolving filter of metric "
                            f"'{metric_name}'. '{path}'"
                        )
            paths[region.key] = resolved

    return paths


def parse_filter(
    value: t.Any,
    dialect: str,
    metric_name: str = "",
    path: t.Any = "",
) -> t.Optional[Filter]:
    """Parse a raw METRIC ``filter`` property value into a :class:`Filter`."""
    if value is None:
        return None

    if isinstance(value, Filter):
        return value

    if isinstance(value, str):
        from sqlmesh.core import dialect as d

        value = d.parse_one(value, dialect=dialect)

    if not isinstance(value, exp.Expr):
        raise ConfigError(
            f"Filter of metric '{metric_name}' must be a SQL boolean expression. '{path}'"
        )

    return Filter(value, dialect=dialect, metric_name=metric_name, path=path)
