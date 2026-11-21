# Definition

Metrics are defined in SQL files in the `metrics/` directory of your SQLMesh project. A single file can contain multiple metric definitions.

A metric is defined with the function `METRIC()` and must include the `name` and `expression` keys. The `name` is case insensitive and must be unique, and the `expression` contains the SQL code used to calculate the metric.

## SQL expression
The expression field can be any SQL statement that contains an aggregation function (e.g., `MAX`, `SUM`). This example uses the `COUNT` function:

```sql linenums="1"
METRIC (
  name unique_account_ids,
  expression COUNT(DISTINCT silver.accounts.account_id)
);
```

All columns referenced in the expression should be fully qualified with the model name. For example, if a model name is `a.b`, a metric referencing column `c` in that model should refer to it as `a.b.c`.

The example above refers to the `account_id` column in model `silver.accounts` using `silver.accounts.account_id`.

### Automatic joins
Metrics can refer to multiple tables and will use the model [grains](../models/overview.md#grain) and [references](../models/overview.md#references) to automatically join them together. `grains` specify a model's key column(s) that uniquely identify the model's rows and `references` specify column(s) that other tables may join to.

For example, a SQLMesh project might contain the `prod.users` and `prod.searches` models with the following `MODEL` DDLs.

The `prod.users` model has a grain of `user_id`, meaning that its rows are uniquely identified by the `user_id` column:

```sql linenums="1"
MODEL (
  name prod.users,
  grain user_id
);
```

The `prod.searches` model specifies `user_id` in its `references` key, signaling that other models may join to its `user_id` column:

```sql linenums="1"
MODEL (
  name prod.searches,
  grain search_id,
  references user_id,
);
```

Because those models specify their grain and references, SQLMesh can correctly generate code for a metric that uses columns from both models.

In this example, `canadian_searchers` sums searches from users located in Canada:

```sql linenums="1"
METRIC (
  name canadian_searchers,
  expression SUM(IF(prod.users.country = 'CAD', prod.searches.num_searches, 0)),
);
```

Because the `prod.users.country` and `prod.searches.num_searches` models have specified their grains/references, SQLMesh can automatically do the correct join between them.

### Derived metrics

Metrics can perform additional operations/calculations with other metrics.

In this example, the third metric `clicks_per_search` is calculated by dividing the first metric `total_searches` by the second metric `total_clicks`:

```sql linenums="1"
METRIC (
  name total_searches,
  expression SUM(prod.searchers.num_searches)
);

METRIC (
  name total_clicks,
  expression SUM(prod.clickers.num_clicks)
);

METRIC (
  name clicks_per_search,
  expression total_clicks / total_searches -- Calculated from the other two metrics
);
```

## Properties

The `METRIC` definition supports the following keys. The `name` and `expression` keys are required.

### name
- The name of the metric. This name is case insensitive and must be unique in a project.

### expression
- A SQL expression consisting of an aggregation, a formula consisting of other metrics, or a combination of both.

### description
- The description of the metric.

### owner
- The owner of the metric. Used for organizational and governance purposes.

### dialect
- The dialect that the expression is written in. It is recommended to leave this empty and rely on the project's default dialect.

### filter
- A boolean SQL expression that determines which fact rows contribute to the metric. The filter is only allowed on **base** metrics (metrics whose expression is a single aggregation over one fact model); derived metrics that reference other metrics cannot declare a filter.
- Every column in the filter must be fully qualified with its model name, for example `demo.tags.label = 'trial'`.
- Supported syntax:
  - column-to-constant comparisons (`=`, `<>`, `<`, `<=`, `>`, `>=`)
  - `IS NULL` / `IS NOT NULL`
  - boolean combinations with `AND`, `OR`, `NOT` and parentheses
- Function calls, arithmetic and column-to-column comparisons are not supported.
- A filter may reference the fact model itself, unique dimensions, and multi-valued related tables (for example tags or activities). Relationships are resolved from the model's `grain` and `references`, including composite join keys.
- Conditions on one related model are evaluated as a unit against a single related row, so conditions from different rows can never be stitched together. When a related object is missing, it is interpreted as an all-NULL row under SQL three-valued logic; `NULL` join keys never match.
- The filter only decides eligibility. Multi-valued tables are pre-aggregated to one boolean flag per fact key, so matching rows never multiply the measure. In a group with no eligible rows, `COUNT` returns `0` and `SUM` returns `NULL`.

#### Filter example

```sql
METRIC (
  name trial_sales,
  expression SUM(demo.sales.amount),
  filter demo.tags.label = 'trial'
);
```

A filter combining two multi-valued tables and a same-row predicate:

```sql
METRIC (
  name trial_promo_sales,
  expression SUM(demo.sales.amount),
  filter demo.tags.label = 'trial'
    AND demo.activities.kind = 'promo'
    AND demo.activities.status = 'attended'
);
```
