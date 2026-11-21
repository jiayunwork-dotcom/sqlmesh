MODEL (
  name demo.tags,
  kind SEED (
    path '../seeds/tags.csv'
  ),
  grain (customer_id, label),
  references customer_id
);
