
-- CLEANING SALES DATA (MySQL)
use sales;

select * from sales_data;

-- 1. Remove duplicate rows using ROW_NUMBER()
-- Disable safe update mode
SET SQL_SAFE_UPDATES = 0;

-- Now run the DELETE
DELETE sd
FROM sales_data sd
JOIN (
    SELECT ORDERNUMBER, PRODUCTCODE, ORDERLINENUMBER, MIN(ORDERDATE) AS first_orderdate
    FROM sales_data
    GROUP BY ORDERNUMBER, PRODUCTCODE, ORDERLINENUMBER
) AS keep_rows
ON sd.ORDERNUMBER = keep_rows.ORDERNUMBER
AND sd.PRODUCTCODE = keep_rows.PRODUCTCODE
AND sd.ORDERLINENUMBER = keep_rows.ORDERLINENUMBER
WHERE sd.ORDERDATE > keep_rows.first_orderdate;

-- Re-enable safe updates (optional)
-- SET SQL_SAFE_UPDATES = 1;


-- 2. Remove rows with missing critical values
DELETE FROM sales_data
WHERE
  ORDERNUMBER IS NULL OR
  QUANTITYORDERED IS NULL OR
  PRICEEACH IS NULL OR
  SALES IS NULL OR
  ORDERDATE IS NULL OR
  CUSTOMERNAME IS NULL;

-- 3. Remove negative or zero sales, price, or quantity
DELETE FROM sales_data
WHERE
  SALES <= 0 OR
  PRICEEACH <= 0 OR
  QUANTITYORDERED <= 0;

-- 4. Trim whitespace in text columns
UPDATE sales_data SET
  CUSTOMERNAME = TRIM(CUSTOMERNAME),
  COUNTRY = TRIM(COUNTRY),
  CONTACTLASTNAME = TRIM(CONTACTLASTNAME),
  CONTACTFIRSTNAME = TRIM(CONTACTFIRSTNAME),
  STATUS = TRIM(STATUS),
  DEALSIZE = TRIM(DEALSIZE);

-- 5. Replace non-critical NULLs with 'Unknown'
UPDATE sales_data
SET ADDRESSLINE2 = 'Unknown'
WHERE ADDRESSLINE2 IS NULL;

UPDATE sales_data
SET ADDRESSLINE3 = 'Unknown'
WHERE ADDRESSLINE3 IS NULL;

UPDATE sales_data
SET ADDRESSLINE4 = 'Unknown'
WHERE ADDRESSLINE4 IS NULL;





-- ADD YEAR, MONTH, QUARTER COLUMNS

-- Check if column exists
ALTER TABLE sales_data 
ADD COLUMN MONTH_ID INT,
ADD COLUMN QTR_ID INT;







-- =======================================================
-- EXPLORATORY DATA ANALYSIS (EDA)
-- =======================================================

-- 1. Total Sales and Number of Orders
SELECT 
  COUNT(*) AS total_orders,
  SUM(SALES) AS total_revenue,
  AVG(SALES) AS avg_order_value
FROM sales_data;

-- 2. Sales by Year
SELECT 
  YEAR_ID,
  SUM(SALES) AS yearly_sales,
  COUNT(*) AS order_count
FROM sales_data
GROUP BY YEAR_ID
ORDER BY YEAR_ID;

-- 3. Sales by Quarter
SELECT 
  YEAR_ID,
  QTR_ID,
  SUM(SALES) AS quarterly_sales
FROM sales_data
GROUP BY YEAR_ID, QTR_ID
ORDER BY YEAR_ID, QTR_ID;

-- 4. Sales by Product Line
SELECT 
  PRODUCTLINE,
  SUM(SALES) AS total_sales,
  SUM(QUANTITYORDERED) AS units_sold,
  AVG(PRICEEACH) AS avg_price
FROM sales_data
GROUP BY PRODUCTLINE
ORDER BY total_sales DESC;

-- 5. Top 10 Customers by Sales
SELECT 
  CUSTOMERNAME,
  SUM(SALES) AS total_spent,
  COUNT(*) AS order_count
FROM sales_data
GROUP BY CUSTOMERNAME
ORDER BY total_spent DESC
LIMIT 10;

-- 6. Sales by Country
SELECT 
  COUNTRY,
  SUM(SALES) AS country_sales,
  COUNT(*) AS order_count
FROM sales_data
GROUP BY COUNTRY
ORDER BY country_sales DESC;

-- 7. Sales by Territory
SELECT 
  TERRITORY,
  SUM(SALES) AS territory_sales,
  COUNT(*) AS order_count
FROM sales_data
GROUP BY TERRITORY
ORDER BY territory_sales DESC;

-- 8. Monthly Sales Trend (Overall)
SELECT 
  YEAR_ID,
  MONTH_ID,
  SUM(SALES) AS monthly_sales
FROM sales_data
GROUP BY YEAR_ID, MONTH_ID
ORDER BY YEAR_ID, MONTH_ID;

-- 9. Order Status Distribution
SELECT 
  STATUS,
  COUNT(*) AS count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM sales_data
GROUP BY STATUS;

-- 10. Deal Size Analysis
SELECT 
  DEALSIZE,
  SUM(SALES) AS total_sales,
  AVG(SALES) AS avg_sale,
  COUNT(*) AS order_count
FROM sales_data
GROUP BY DEALSIZE;

-- 11. Top Selling Products (by Units and Revenue)
SELECT 
  PRODUCTCODE,
  SUM(QUANTITYORDERED) AS total_quantity_sold,
  SUM(SALES) AS total_revenue
FROM sales_data
GROUP BY PRODUCTCODE
ORDER BY total_quantity_sold DESC
LIMIT 10;

-- 12. Average Sales per Month by Year
SELECT 
  YEAR_ID,
  AVG(monthly_sales) AS avg_monthly_sales
FROM (
  SELECT 
    YEAR_ID, 
    MONTH_ID, 
    SUM(SALES) AS monthly_sales
  FROM sales_data
  GROUP BY YEAR_ID, MONTH_ID
) AS monthly_summary
GROUP BY YEAR_ID;

select * from sales_data;




