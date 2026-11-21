from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.dialects.dialect import DialectType
from sqlglot.optimizer import Scope, find_all_in_scope, optimize
from sqlglot.optimizer.optimize_joins import optimize_joins
from sqlglot.optimizer.qualify import qualify

from sqlmesh.core import dialect as d
from sqlmesh.core.metric.definition import Metric, remove_namespace
from sqlmesh.core.metric.filter import (
    FilterRegion,
    validate_columns,
)
from sqlmesh.utils.errors import SQLMeshError

if t.TYPE_CHECKING:
    from sqlmesh.core.reference import ReferenceGraph


# One aggregation per (fact model, base metric name, filter SQL). Using the
# metric name as part of the identity keeps distinct populations of the same
# measure (filtered vs. unfiltered) separate even when their raw aggregation
# SQL is identical.
AggKey = t.Tuple[str, str, t.Optional[str]]

SourceAggsAndJoins = t.Dict[
    str,
    t.Tuple[t.Dict[AggKey, t.Tuple[exp.AggFunc, str]], t.Dict[str, t.Optional[exp.Join]]],
]


class Rewriter:
    def __init__(
        self,
        graph: ReferenceGraph,
        metrics: t.Dict[str, Metric],
        dialect: DialectType = "",
        join_type: str = "FULL",
        semantic_schema: str = "__semantic",
        semantic_table: str = "__table",
    ):
        self.graph = graph
        self.metrics = metrics
        self.dialect = dialect
        self.join_type = join_type
        self.semantic_name = f"{semantic_schema}.{semantic_table}"

    def rewrite(self, expression: exp.Expr) -> exp.Expr:
        # Snapshot the selects before any replacement: the NULL-group merge
        # wrapper introduces new SELECT nodes and must not be revisited.
        selects = list(expression.find_all(exp.Select))
        root_replacement: t.Optional[exp.Expr] = None
        for select in selects:
            replacement = self._expand(select)
            if replacement is not None and replacement is not select:
                if select.parent is None:
                    # A root expression has no parent to replace into.
                    root_replacement = replacement
                else:
                    select.replace(replacement)
            target = replacement or select
            # Metric formula expansion turns references like ``a / b`` into
            # concrete divisions; guard every divisor with NULLIF so a zero or
            # NULL divisor yields NULL instead of a runtime error. Mutating
            # during traversal is unsafe, so collect the divisions first.
            for div in list(target.find_all(exp.Div)):
                denominator = div.expression
                # The metric expansion already protects divisions with NULLIF;
                # only guard raw divisions that do not have one yet.
                if not isinstance(denominator, exp.Nullif):
                    div.set(
                        "expression",
                        exp.Nullif(
                            this=denominator.copy(),
                            expression=exp.Literal.number(0),
                        ),
                    )

        return root_replacement or expression

    def _build_sources(self, projections: t.List[exp.Expr]) -> SourceAggsAndJoins:
        """Collect every distinct (measure model, base metric, filter) triple."""
        sources: SourceAggsAndJoins = {}

        for projection in projections:
            for ref in find_all_in_scope(projection, d.MetricAgg):
                metric = self.metrics[ref.this.name]
                ref.replace(metric.formula.this)

                # Walk every base aggregation reachable from this metric. A
                # derived metric may reference several base metrics; each keeps
                # its own filter identity.
                for base_name, base_metric in self._base_metrics(metric).items():
                    filter_sql = (
                        base_metric.filter.sql() if base_metric.filter else None
                    )
                    for agg, (measure, dims) in base_metric.aggs.items():
                        aggs, joins = sources.setdefault(measure, ({}, dict()))
                        key: AggKey = (measure, base_name, filter_sql)
                        if key not in aggs:
                            aggs[key] = (agg, base_name)
                        for dim in dims:
                            joins[dim] = None

        return sources

    def _base_metrics(self, metric: Metric) -> t.Dict[str, Metric]:
        """All base metrics feeding ``metric``, including itself when base."""
        result: t.Dict[str, Metric] = {}

        def visit(candidate: Metric) -> None:
            has_ref = any(
                isinstance(node, exp.Column) and not node.table
                for node in candidate.expression.walk()
            )
            if not has_ref:
                result[candidate.name] = candidate
                return
            for node in candidate.expression.find_all(exp.Column):
                if not node.table and node.name in self.metrics:
                    visit(self.metrics[node.name])

        visit(metric)
        return result

    def _expand(self, select: exp.Select) -> t.Optional[exp.Expr]:
        base = select.args["from_"].this.find(exp.Table)
        base_alias = base.alias_or_name
        base_name = exp.table_name(base)

        sources: SourceAggsAndJoins = (
            {} if base_name == self.semantic_name else {base_name: ({}, {})}
        )
        sources.update(self._build_sources(select.selects))

        group = select.args.pop("group", None)
        group_by = group.expressions if group else []

        mapping = {
            remove_namespace(exp.table_name(source.assert_is(exp.Table))): name
            for name, source in Scope(select).references
            if name != base_alias
        }

        explicit_joins = {exp.table_name(join.this): join for join in select.args.pop("joins", [])}
        where = select.args.pop("where", None)

        # Keep the original grouping expressions (with the user's aliases) for
        # the FULL JOIN predicates: the per-fact copies get requalified inside
        # each fact subquery and must not leak into the join clause.
        original_group_by = [e.copy() for e in group_by]

        built_sources: t.List[
            t.Tuple[exp.Select, t.List[exp.Expr], str, str, bool]
        ] = []
        for name, (aggs, joins) in sources.items():
            source: exp.Expr = exp.to_table(name)
            table_name = remove_namespace(name)

            if not isinstance(source, exp.Select):
                source = exp.Select().from_(
                    exp.alias_(source, table_name, table=True, copy=False), copy=False
                )

            joins.update(explicit_joins)

            query, group_projections, has_eligibility = self._build_fact_subquery(
                source=source,
                fact_name=name,
                fact_alias=table_name,
                aggs=aggs,
                joins=joins,
                group_by=group_by,
                mapping=mapping,
                where=where,
                base_alias=base_alias,
            )

            if not query.selects:
                query.select("*", copy=False)

            built_sources.append(
                (query, group_projections, table_name, name, has_eligibility)
            )

        any_eligibility = any(item[4] for item in built_sources)

        for i, (query, group_projections, table_name, _name, _elig) in enumerate(
            built_sources
        ):
            # Materialized group keys are only needed for the explicit
            # LEFT/RIGHT UNION construction used when eligibility joins are
            # present.
            if group_projections and any_eligibility:
                self._add_join_keys(query, group_projections, table_name)

            if i == 0:
                select.from_(query.subquery(base_alias, copy=False), copy=False)
            elif not group_projections:
                # Ungrouped single-row aggregations: every fact produces one
                # row, so a plain cross join preserves the values without any
                # group key handling.
                select.join(
                    query.subquery(table_name, copy=False),
                    on=None,
                    join_type="CROSS",
                    join_alias=table_name,
                    copy=False,
                )
            elif not any_eligibility:
                # No eligibility LEFT JOINs are involved: keep the legacy FULL
                # JOIN directly on the grouping expressions, which the engine
                # merges correctly for ordinary fact subqueries.
                on_conditions = []
                for group_expression in original_group_by:
                    left = group_expression.copy()
                    right = group_expression.copy()
                    for column in right.find_all(exp.Column):
                        column.set("table", exp.to_identifier(table_name))
                    on_conditions.append(left.eq(right))
                select.join(
                    query,
                    on=on_conditions,
                    join_type=self.join_type,
                    join_alias=table_name,
                    copy=False,
                )
            else:
                first_table_name = built_sources[0][2]
                # DuckDB (and similar engines) do not reliably merge NULL
                # groups in a FULL JOIN once the fact subqueries contain
                # eligibility LEFT JOINs. Build FULL OUTER semantics explicitly
                # with a LEFT branch (all base groups) and a RIGHT branch
                # restricted to groups absent on the left, UNION ALL'd together.
                # Materialized sentinel keys make every predicate a plain
                # non-NULL equi-join, so the unmatched test is exact.
                on_conditions = [
                    exp.column(
                        f"__group_key_{j}_{first_table_name}", table=base_alias
                    ).eq(
                        exp.column(
                            f"__group_key_{j}_{table_name}", table=table_name
                        )
                    )
                    for j in range(len(group_projections))
                ]
                base_from = select.args["from_"].this
                # Both branches must expose the same column list so UNION ALL
                # type-checks: every user projection plus the internal group
                # keys from both sides.
                branch_projections = [p.copy() for p in select.selects]
                for j in range(len(group_projections)):
                    branch_projections.append(
                        exp.column(
                            f"__group_key_{j}_{first_table_name}", table=base_alias
                        )
                    )
                    branch_projections.append(
                        exp.column(
                            f"__group_key_{j}_{table_name}", table=table_name
                        )
                    )

                left = (
                    exp.Select()
                    .select(*branch_projections, copy=False)
                    .from_(base_from.copy(), copy=False)
                    .join(
                        query.subquery(table_name, copy=False),
                        on=on_conditions,
                        join_type="LEFT",
                        join_alias=table_name,
                        copy=False,
                    )
                )
                right = (
                    exp.Select()
                    .select(*[p.copy() for p in branch_projections], copy=False)
                    .from_(base_from.copy(), copy=False)
                    .join(
                        query.subquery(table_name, copy=False),
                        on=[c.copy() for c in on_conditions],
                        join_type="RIGHT",
                        join_alias=table_name,
                        copy=False,
                    )
                )
                # A right-only row has a NULL base-side group key.
                base_first_key = on_conditions[0].left
                right.where(base_first_key.is_(exp.Null()), copy=False)

                union = exp.Union(
                    this=left,
                    expression=right,
                    distinct=False,
                )
                select.set(
                    "from_",
                    exp.From(this=exp.Subquery(this=union, alias="_unioned_facts")),
                )
                # The outer projection referenced the first fact's alias; it
                # must now read from the unioned subquery instead.
                for projection in select.expressions:
                    for column in projection.find_all(exp.Column):
                        if column.table == base_alias:
                            column.set("table", None)

            for node in find_all_in_scope(query, exp.Column, exp.TableAlias):  # type: ignore[arg-type,var-annotated]
                if isinstance(node, exp.Column):
                    if node.table in mapping:
                        node.set("table", exp.to_identifier(mapping[node.table]))
                else:
                    if node.name in mapping:
                        node.set("this", exp.to_identifier(mapping[node.name]))

        # NULL groups from different facts merge through the materialized
        # sentinel join keys (see _add_join_keys). No extra outer merge is
        # needed because the keys are real non-NULL columns in both fact
        # subqueries.
        return None

    def _build_fact_subquery(
        self,
        source: exp.Select,
        fact_name: str,
        fact_alias: str,
        aggs: t.Dict[AggKey, t.Tuple[exp.AggFunc, str]],
        joins: t.Dict[str, t.Optional[exp.Join]],
        group_by: t.List[exp.Expr],
        mapping: t.Dict[str, str],
        where: t.Optional[exp.Where],
        base_alias: str,
    ) -> t.Tuple[exp.Select, t.List[exp.Expr], bool]:
        """Build the aggregation subquery of one fact model.

        Eligibility of multi-valued related models is computed in dedicated
        subqueries (one row per distinct fact key) and LEFT JOINed back, so a
        fact row keeps exactly its original multiplicity regardless of how many
        related rows match.

        Returns the query, its grouping projections, and whether any
        eligibility join was added.
        """
        grain = [e.copy() for e in group_by]
        reverse_mapping = {v: k for k, v in mapping.items()}
        resolved_fact = self.graph._resolve_model_name(fact_name) or fact_name

        for expr in grain:
            for node in expr.walk():
                if isinstance(node, exp.Column):
                    models = self.graph.models_for_column(fact_name, node.name)

                    if resolved_fact in models:
                        node.args["table"] = exp.to_identifier(fact_alias)
                    elif models:
                        t = reverse_mapping.get(node.table)
                        model = next(
                            (model for model in models if remove_namespace(model) == t),
                            models[0],
                        )
                        node.args["table"] = exp.to_identifier(t or remove_namespace(model))
                        if model not in joins:
                            joins[model] = None

        # One eligibility plan per related model; flags are shared across all
        # filtered metrics of this fact that touch the same model. Every region
        # predicate is registered once so the same predicate never produces two
        # BOOL_OR columns even when it appears in several metric filters.
        region_plans: t.Dict[str, RegionPlan] = {}
        wrapped: t.Dict[AggKey, exp.AggFunc] = {}

        for key, (agg, base_name) in aggs.items():
            filter_sql = key[2]
            if filter_sql is None:
                wrapped[key] = agg
                continue

            metric = self.metrics[base_name]
            assert metric.filter is not None
            assert metric.filter.sql() == filter_sql

            model_aliases = self._resolve_filter_columns(metric, fact_name)
            flag_columns: t.Dict[t.Tuple[str, str], exp.Column] = {}

            for region in metric.filter.regions:
                resolved_region_model = (
                    self.graph._resolve_model_name(region.model) or region.model
                )
                if resolved_region_model == resolved_fact:
                    continue
                plan = self._region_plan(fact_name, region, region_plans)
                flag_columns[region.key] = exp.column(
                    plan.flag_alias(region), table=plan.join_alias
                )

            eligibility = metric.filter.rewrite_with_regions(
                fact_model=fact_name,
                graph=self.graph,
                model_aliases=model_aliases,
                region_flags=flag_columns,
            )
            wrapped[key] = _wrap_agg_with_eligibility(agg, eligibility)

        # Regular measure/group joins (subject to the cardinality rules).
        query = self._add_joins(source, fact_name, joins, grain, mapping)

        # Eligibility joins on top of the regular joins. Each returns one row
        # per fact key, so fact multiplicity is preserved.
        for plan in region_plans.values():
            query.join(
                plan.cte.subquery(plan.join_alias, copy=False),
                on=plan.join_on(fact_alias),
                join_type="LEFT",
                copy=False,
            )

        query.select(*sorted(wrapped.values(), key=str), copy=False)

        if where:
            query.where(
                _replace_table(where.this.copy(), fact_alias, base_alias), copy=False
            )

        group_projections = query.selects[: len(grain)]
        return query, group_projections, bool(region_plans)

    def _resolve_filter_columns(self, metric: Metric, fact_name: str) -> t.Dict[str, str]:
        """Validate filter columns/paths and return model -> fact-scope aliases."""
        assert metric.filter is not None
        validate_columns(
            metric.filter,
            graph=self.graph,
            fact_model=fact_name,
            metric_name=metric.name,
            path=getattr(metric, "_path", ""),
        )
        return {model: remove_namespace(model) for model in metric.filter.models}

    def _region_plan(
        self,
        fact_name: str,
        region: FilterRegion,
        cache: t.Dict[str, RegionPlan],
    ) -> "RegionPlan":
        """Build (or reuse) the eligibility plan for one related model."""
        graph_target = self.graph._resolve_model_name(region.model) or region.model
        plan = cache.get(graph_target)
        if plan is None:
            graph_fact = self.graph._resolve_model_name(fact_name) or fact_name
            path = self.graph.find_eligibility_path(graph_fact, graph_target)
            plan = RegionPlan(
                fact_name=fact_name,
                graph_fact=graph_fact,
                graph_model=graph_target,
                qualified_model=region.model,
                path=path,
                dialect=self.dialect,
            )
            cache[graph_target] = plan
        plan.add_region(region)
        return plan

    def _add_join_keys(
        self,
        subquery: exp.Select,
        group_projections: t.List[exp.Expr],
        subquery_alias: str,
    ) -> t.List[str]:
        """Project bare NULL-safe join key columns into a fact subquery.

        Each grouping expression gets an extra alias whose value is the group
        expression cast to TEXT with NULL replaced by a sentinel. FULL JOINs
        then compare those bare columns with plain ``=`` so NULL groups from
        different facts merge into one row.
        """
        names = []
        for i, group_expression in enumerate(group_projections):
            name = f"__group_key_{i}_{subquery_alias}"
            expression = _qualify_group_expression(group_expression, subquery_alias)
            key = exp.alias_(
                exp.Coalesce(
                    this=exp.Cast(this=expression, to=exp.DataType.build("TEXT")),
                    expressions=[exp.Literal.string(_NULL_GROUP_SENTINEL)],
                ),
                name,
            )
            subquery.select(key, copy=False)
            names.append(name)
        return names

    def _add_joins(
        self,
        source: exp.Select,
        name: str,
        joins: t.Dict[str, t.Optional[exp.Join]],
        group_by: t.List[exp.Expr],
        mapping: t.Dict[str, str],
    ) -> exp.Select:
        table_name = remove_namespace(name)
        reverse_mapping = {v: k for k, v in mapping.items()}
        catalog = name.split(".")[0] if name.count(".") >= 2 else ""

        def qualify_model(graph_name: str) -> str:
            return f"{catalog}.{graph_name}" if catalog else graph_name

        for target, join in joins.items():
            # The fact model itself never needs a self-join for measure work.
            if (self.graph._resolve_model_name(target) or target) == (
                self.graph._resolve_model_name(name) or name
            ):
                continue

            path = self.graph.find_path(name, target)
            for i in range(len(path) - 1):
                a_ref = path[i]
                b_ref = path[i + 1]
                a_model_alias = remove_namespace(a_ref.model_name)
                b_model_alias = remove_namespace(b_ref.model_name)

                a = a_ref.expression.copy()
                a.set("table", exp.to_identifier(a_model_alias))
                b = b_ref.expression.copy()
                b.set("table", exp.to_identifier(b_model_alias))
                on = a.eq(b)

                if join:
                    join.set("on", on)
                    source.append("joins", join)
                else:
                    source.join(
                        qualify_model(b_ref.model_name),
                        on=on,
                        join_type="LEFT",
                        join_alias=b_model_alias,
                        dialect=self.dialect,
                        copy=False,
                    )

        return source.select(*group_by, copy=False).group_by(*group_by, copy=False)


class RegionPlan:
    """Eligibility subquery and LEFT JOIN for one related model.

    Groups the related model's rows back onto the fact-side key of the
    reference path, producing one row per distinct fact key with one boolean
    flag per region predicate.  Fan-out of the (possibly non-unique) final hop
    is collapsed by ``GROUP BY`` so a fact row can never be multiplied.
    """

    def __init__(
        self,
        fact_name: str,
        graph_fact: str,
        graph_model: str,
        qualified_model: str,
        path: "ReferencePath",
        dialect: DialectType = "",
    ):
        self.fact_name = fact_name
        self.graph_fact = graph_fact
        self.graph_model = graph_model
        self.qualified_model = qualified_model
        self.path = path
        self.dialect = dialect

        self.model_alias = remove_namespace(graph_model)
        self.join_alias = f"__elig_join_{self.model_alias}"
        self.fact_alias = remove_namespace(graph_fact)
        # The flag subquery joins back onto the outer fact subquery, whose
        # table alias keeps the full catalog prefix flattened in.
        self.outer_fact_alias = remove_namespace(self.fact_name)
        self.fact_key_columns = path[0].columns if path else []
        self._flags: t.Dict[t.Tuple[str, str], str] = {}
        self.cte = self._build_cte()

    def add_region(self, region: FilterRegion) -> None:
        if region.key in self._flags:
            return

        # Every distinct region predicate gets its own BOOL_OR flag. A larger
        # predicate (e.g. ``A AND B``) cannot be reused for a smaller one
        # (``A``): same-row semantics make their truth sets different.
        flag_name = f"__eligible_{self.model_alias}_{len(self._flags)}"
        self._flags[region.key] = flag_name
        predicate = region.predicate(self.model_alias)
        self.cte.select(
            exp.alias_(exp.LogicalOr(this=predicate), flag_name), copy=False
        )

    def flag_alias(self, region: FilterRegion) -> str:
        return self._flags[region.key]

    def join_on(self, fact_alias: str) -> exp.Condition:
        conditions = []
        for column in self.fact_key_columns:
            conditions.append(
                exp.column(column, table=fact_alias).eq(
                    exp.column(column, table=self.join_alias)
                )
            )
        return exp.and_(*conditions, copy=False)

    def _build_cte(self) -> exp.Select:
        if not self.path:
            raise SQLMeshError(
                f"Eligibility path from '{self.graph_fact}' to '{self.graph_model}' is empty"
            )

        # Use a distinct inner fact alias so the subquery never clashes with
        # the outer fact's alias.
        inner_fact_alias = f"__elig_src_{self.fact_alias}"
        query = exp.Select().from_(
            exp.alias_(exp.to_table(self.fact_name), inner_fact_alias, table=True),
            copy=False,
        )

        # Walk every hop, including the possibly non-unique final one. The
        # GROUP BY below collapses any fan-out.
        catalog = self.fact_name.split(".")[0] if "." in self.fact_name else ""
        for i in range(len(self.path) - 1):
            a_ref = self.path[i]
            b_ref = self.path[i + 1]
            a_alias = (
                inner_fact_alias
                if i == 0
                else remove_namespace(a_ref.model_name)
            )
            b_alias = remove_namespace(b_ref.model_name)
            target_table = (
                f"{catalog}.{b_ref.model_name}" if catalog else b_ref.model_name
            )
            query.join(
                target_table,
                on=_ref_eq(a_ref, a_alias, b_ref, b_alias),
                join_type="LEFT",
                join_alias=b_alias,
                dialect=self.dialect,
                copy=False,
            )

        group_exprs = []
        for column in self.fact_key_columns:
            key = exp.column(column, table=inner_fact_alias)
            query.select(key, copy=False)
            group_exprs.append(key.copy())

        return query.group_by(*group_exprs, copy=False)


def _ref_eq(a_ref: t.Any, a_alias: str, b_ref: t.Any, b_alias: str) -> exp.Condition:
    a = a_ref.expression.copy()
    a.set("table", exp.to_identifier(a_alias))
    b = b_ref.expression.copy()
    b.set("table", exp.to_identifier(b_alias))
    return a.eq(b)


def _wrap_agg_with_eligibility(agg: exp.AggFunc, eligibility: exp.Expr) -> exp.AggFunc:
    """Wrap an aggregation's measure so it only accumulates eligible rows.

    Produces ``SUM(IF(eligibility, measure, NULL))`` or
    ``COUNT(IF(eligibility, measure, NULL))``; the eligibility predicate stays
    inside the aggregate so it can reference columns of joined eligibility
    subqueries without appearing in the GROUP BY.

    When no row of a group is eligible the IF yields only NULLs: COUNT becomes
    0 and SUM becomes NULL.
    """
    agg = agg.copy()
    # Metric aggregations are usually wrapped in an Alias (SUM(x) AS name);
    # mutate the aggregate itself, keeping the alias untouched.
    target = agg.this if isinstance(agg, exp.Alias) else agg

    if isinstance(target, exp.Count) and isinstance(target.this, exp.Distinct):
        distinct = target.this
        inner = distinct.this
        distinct.set(
            "this",
            exp.If(this=eligibility.copy(), true=inner.copy(), false=exp.Null()),
        )
        return agg

    measure = target.this
    target.set(
        "this",
        exp.If(this=eligibility.copy(), true=measure.copy(), false=exp.Null()),
    )
    return agg


def _replace_table(node: exp.Expr, table: str, base_alias: str) -> exp.Expr:
    for column in find_all_in_scope(node, exp.Column):
        if column.table == base_alias:
            column.args["table"] = exp.to_identifier(table)
    return node


def _qualify_group_expression(group_expression: exp.Expr, alias: str) -> exp.Expr:
    expression = group_expression.copy()
    for column in expression.find_all(exp.Column):
        column.set("table", exp.to_identifier(alias))
    return expression


_NULL_GROUP_SENTINEL = "__sqlmesh_null_group__"


def rewrite(
    sql: str | exp.Expr,
    graph: ReferenceGraph,
    metrics: t.Dict[str, Metric],
    dialect: t.Optional[str] = "",
) -> exp.Expr:
    rewriter = Rewriter(graph=graph, metrics=metrics, dialect=dialect)

    return optimize(
        d.parse_one(sql, dialect=dialect) if isinstance(sql, str) else sql,
        dialect=dialect,
        quote_identifiers=False,
        rules=(
            qualify,
            rewriter.rewrite,
            optimize_joins,
        ),
    )
