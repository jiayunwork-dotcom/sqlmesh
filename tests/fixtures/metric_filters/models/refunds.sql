MODEL (
  name demo.refunds,
  kind SEED (
    path '../seeds/refunds.csv'
  ),
  grain refund_id,
  references (customer_id, sale_id)
);
