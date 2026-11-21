MODEL (
  name demo.activities,
  kind SEED (
    path '../seeds/activities.csv'
  ),
  grain (customer_id, kind, status),
  references customer_id
);
