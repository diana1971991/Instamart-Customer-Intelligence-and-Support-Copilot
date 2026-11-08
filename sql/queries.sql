-- Q1 — WHERE then HAVING

SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE order_amount > 500 AND status = 'completed'
GROUP BY customer_id
HAVING total_spend > 8000;

--=============================================================================

-- Q2: Three-table join (Revenue by city and product category)
SELECT c.city, p.category, SUM(o.order_amount) AS total_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

--=============================================================================

-- Q3: Scalar subquery (Products priced above the average catalogue price)
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

--=============================================================================

-- Q4: LEFT JOIN (Customers who have no completed order)
SELECT c.customer_id
FROM customers c
LEFT JOIN (SELECT DISTINCT customer_id FROM orders WHERE status = 'completed') o 
  ON c.customer_id = o.customer_id
WHERE o.customer_id IS NULL;

--=============================================================================

-- Q5: Common Table Expression (Customers whose completed spend is above the average of all customer totals)
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
overall_avg AS (
    SELECT AVG(total_spend) AS avg_total_spend FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend, ROUND(oa.avg_total_spend, 2) AS benchmark_avg
FROM customer_spend cs, overall_avg oa
WHERE cs.total_spend > oa.avg_total_spend;

--=============================================================================

-- Q6: Ranking window functions (Rank, Dense Rank, Row Number partitioned by city ordered by total spend desc)
WITH customer_city_spend AS (
    SELECT c.city, o.customer_id, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city, customer_id, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) as row_num
FROM customer_city_spend;

--=============================================================================

-- Q7: Running total (Daily completed cumulative revenue)
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS daily_rev
    FROM orders
    WHERE status = 'completed'
    GROUP BY day
)
SELECT day, daily_rev,
       SUM(daily_rev) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
FROM daily_revenue;