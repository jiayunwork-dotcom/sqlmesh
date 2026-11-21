MODEL (
  name demo.customers,
  kind SEED (
    path '../seeds/customers.csv'
  ),
  grain customer_id
);
