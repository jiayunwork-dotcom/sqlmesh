-- Unfiltered baseline over the sales fact.
METRIC (
  name total_sales,
  expression SUM(metric_filters.demo.sales.amount)
);

METRIC (
  name order_count,
  expression COUNT(metric_filters.demo.sales.sale_id)
);

METRIC (
  name distinct_sale_customers,
  expression COUNT(DISTINCT metric_filters.demo.sales.customer_id)
);

-- Customers carrying the 'trial' tag. The tag table has repeated rows; the
-- filter only determines eligibility and must never multiply the amount.
METRIC (
  name trial_sales,
  expression SUM(metric_filters.demo.sales.amount),
  filter metric_filters.demo.tags.label = 'trial'
);

METRIC (
  name trial_order_count,
  expression COUNT(metric_filters.demo.sales.sale_id),
  filter metric_filters.demo.tags.label = 'trial'
);

-- Customers that attended a promo activity. Same-row semantics: kind and
-- status must match on one activity record.
METRIC (
  name promo_attendee_sales,
  expression SUM(metric_filters.demo.sales.amount),
  filter metric_filters.demo.activities.kind = 'promo'
    AND metric_filters.demo.activities.status = 'attended'
);

-- Two multi-valued tables used together: tags and activities. Eligibility is
-- evaluated independently and must not produce a cartesian product.
METRIC (
  name trial_promo_sales,
  expression SUM(metric_filters.demo.sales.amount),
  filter metric_filters.demo.tags.label = 'trial'
    AND metric_filters.demo.activities.kind = 'promo'
    AND metric_filters.demo.activities.status = 'attended'
);

-- OR mixing a fact-side condition with a related-table condition. A sale with
-- no activity still qualifies through its own region (three valued logic).
METRIC (
  name west_or_promo_sales,
  expression SUM(metric_filters.demo.sales.amount),
  filter metric_filters.demo.sales.region = 'WEST'
    OR (
      metric_filters.demo.activities.kind = 'promo'
      AND metric_filters.demo.activities.status = 'attended'
    )
);

-- Counterexample for same-row semantics: customer 2 has one webinar activity
-- (status attended) and one promo activity (status invited). Stitching halves
-- across rows would wrongly qualify them for "webinar AND invited"; only
-- customer 4 has a single row satisfying both predicates.
METRIC (
  name half_condition_sales,
  expression SUM(metric_filters.demo.sales.amount),
  filter metric_filters.demo.activities.kind = 'webinar'
    AND metric_filters.demo.activities.status = 'invited'
);

-- COUNT variant of the promo-attendee population, used to verify the
-- empty-group rule: COUNT returns 0 where SUM returns NULL.
METRIC (
  name promo_order_count,
  expression COUNT(metric_filters.demo.sales.sale_id),
  filter metric_filters.demo.activities.kind = 'promo'
    AND metric_filters.demo.activities.status = 'attended'
);

-- Refund measures on a different fact model.
METRIC (
  name total_refunds,
  expression SUM(metric_filters.demo.refunds.amount)
);

METRIC (
  name trial_refunds,
  expression SUM(metric_filters.demo.refunds.amount),
  filter metric_filters.demo.tags.label = 'trial'
);

-- Derived metrics read the aggregate values; filters live only on base metrics.
METRIC (
  name trial_sales_share,
  expression trial_sales / total_sales
);

METRIC (
  name refund_rate,
  expression total_refunds / total_sales
);

-- Ratio of two populations: in groups with no promo attendees the numerator is
-- 0 and in groups with neither population the divisor is 0, which must yield
-- NULL rather than an error.
METRIC (
  name promo_to_trial_ratio,
  expression promo_order_count / trial_order_count
);

-- Numerator 0 and divisor 0 inside the WEST group: the half-condition
-- population is empty there while the promo population is empty too, giving
-- a 0 / 0 division that must evaluate to NULL instead of raising.
METRIC (
  name empty_to_empty_ratio,
  expression half_condition_sales / promo_attendee_sales
);
