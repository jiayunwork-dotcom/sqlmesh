MODEL (
  name demo.sales,
  kind SEED (
    path '../seeds/sales.csv'
  ),
  grain sale_id,
  references customer_id
);
