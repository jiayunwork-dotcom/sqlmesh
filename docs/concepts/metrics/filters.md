# Metric filters

Metric filters let a base metric describe a *population* independently of the
measure it aggregates. A filtered metric answers questions like "sales of
customers who attended a promo" without joining the related table into the
fact query by hand.

## What a filter is

A filter is a boolean expression attached to a `METRIC` definition:

```sql
METRIC (
  name trial_sales,
  expression SUM(demo.sales.amount),
  filter demo.tags.label = 'trial'
);
```

The filter decides whether each fact row is eligible; the aggregation itself
is unchanged. The same measure can therefore be reused across several
populations, and derived metrics can combine the resulting aggregates:

```sql
METRIC (
  name trial_sales_share,
  expression trial_sales / total_sales
);
```

Filters are only valid on base metrics containing one `SUM`, `COUNT` or
`COUNT(DISTINCT)` over a single fact model. A derived metric that references
other metrics cannot declare its own filter; the populations it reads are
those of its base metrics.

## Supported syntax

* Column-to-constant comparisons: `=`, `<>`, `<`, `<=`, `>`, `>=`.
* `IS NULL` and `IS NOT NULL`.
* Boolean combinations with `AND`, `OR`, `NOT` and parentheses.

Every column must be fully qualified by its model name. Function calls,
arithmetic inside predicates and column-to-column comparisons are rejected at
load time with a locatable error.

## How eligibility is computed

Related tables can be multi-valued: one customer may have many tags or
activity rows. The rewriter never joins such a table directly into the fact
aggregation. Instead, for each related model it builds an eligibility
subquery:

1. Join the fact keys to the related model along the reference path.
2. Evaluate the region predicate for that model.
3. Collapse all matching rows to one boolean per fact key with `BOOL_OR`.
4. LEFT JOIN the resulting one-row-per-key result back to the fact.

The fact table's row count is therefore identical with or without the filter.
Repeated tags cannot multiply an amount, and two multi-valued tables used
together cannot produce a cartesian product.

### Same-row semantics

All conditions on one related model form one region and are evaluated against
a single related row. Given activities:

| customer_id | kind    | status   |
| ----------- | ------- | -------- |
| 2           | promo   | invited  |
| 2           | webinar | attended |

the filter `kind = 'webinar' AND status = 'invited'` does **not** qualify
customer 2, because no single row satisfies both halves.

### Missing objects and NULL keys

A fact row with no matching related object behaves as if joined to an
all-NULL row, so its region predicates evaluate to NULL under standard
three-valued logic. An `OR` with a fact-side condition still admits the row;
an `AND` excludes it. Join keys that are themselves NULL never match, because
the join uses SQL equality rather than null-safe matching.

## Groups, empty populations and zero divisors

* Grouping values present on only one fact still appear in the combined
  result, and NULL group values from different facts merge into one group.
* When no row of a group is eligible, `COUNT` returns `0` and `SUM` returns
  `NULL`.
* Division in derived metrics is protected with `NULLIF(..., 0)`, so a zero
  divisor yields NULL instead of a runtime error.
* A `WHERE` clause on the semantic query applies to every fact input; a
  metric-level filter cannot be promoted into a shared `WHERE` clause.

## Rejection rules

The rewriter fails with a clear error, rather than guessing, when:

* a filter column does not exist on its declared model;
* no reference path connects the fact to a filtered model;
* several equally short, equally unique paths make the relationship ambiguous;
* a non-unique (many-to-many) hop would be needed anywhere except the final
  hop of an eligibility path;
* a filtered base metric uses an unsupported aggregation or measures from more
  than one model.
