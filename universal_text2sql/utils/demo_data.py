"""Utility to seed a demo SQLite database with realistic sample data.

Used when no ``DATABASE_URL`` is configured so that first-time users can
immediately try the agent without setting up an external database.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from universal_text2sql.database.connector import DatabaseConnector

logger = logging.getLogger(__name__)

_SEED_SQL = """
-- Customers
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    country       TEXT NOT NULL,
    created_at    DATE NOT NULL
);

INSERT OR IGNORE INTO customers VALUES
  (1,  'Alice',   'Smith',    'alice@example.com',   'USA',    '2022-01-15'),
  (2,  'Bob',     'Jones',    'bob@example.com',     'UK',     '2022-03-20'),
  (3,  'Carol',   'White',    'carol@example.com',   'Canada', '2022-05-10'),
  (4,  'David',   'Brown',    'david@example.com',   'USA',    '2022-07-08'),
  (5,  'Eve',     'Taylor',   'eve@example.com',     'Germany','2022-09-01'),
  (6,  'Frank',   'Wilson',   'frank@example.com',   'France', '2023-01-12'),
  (7,  'Grace',   'Martin',   'grace@example.com',   'USA',    '2023-03-18'),
  (8,  'Henry',   'Anderson', 'henry@example.com',   'UK',     '2023-05-22'),
  (9,  'Iris',    'Thomas',   'iris@example.com',    'Japan',  '2023-07-30'),
  (10, 'Jack',    'Jackson',  'jack@example.com',    'Australia','2023-09-15');

-- Products
CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    price         REAL NOT NULL,
    stock         INTEGER NOT NULL
);

INSERT OR IGNORE INTO products VALUES
  (1,  'Laptop Pro',      'Electronics', 1299.99, 50),
  (2,  'Wireless Mouse',  'Electronics',   29.99, 200),
  (3,  'Desk Chair',      'Furniture',    299.99, 30),
  (4,  'Standing Desk',   'Furniture',    499.99, 15),
  (5,  'Notebook',        'Stationery',     4.99, 500),
  (6,  'Pen Set',         'Stationery',     9.99, 300),
  (7,  'Coffee Maker',    'Appliances',    89.99, 75),
  (8,  'Headphones',      'Electronics',  149.99, 120),
  (9,  'Monitor 27"',     'Electronics',  379.99, 40),
  (10, 'Keyboard',        'Electronics',   79.99, 180);

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    DATE NOT NULL,
    status        TEXT NOT NULL,
    total_amount  REAL NOT NULL
);

INSERT OR IGNORE INTO orders VALUES
  (1,  1, '2023-01-10', 'completed', 1329.98),
  (2,  2, '2023-02-14', 'completed',  299.99),
  (3,  3, '2023-03-05', 'shipped',    529.98),
  (4,  4, '2023-03-20', 'completed',   89.99),
  (5,  5, '2023-04-15', 'cancelled',  149.99),
  (6,  1, '2023-05-01', 'completed',   79.99),
  (7,  6, '2023-05-22', 'completed',  379.99),
  (8,  7, '2023-06-10', 'shipped',    499.99),
  (9,  8, '2023-07-04', 'completed',   29.99),
  (10, 9, '2023-08-18', 'completed',  459.98),
  (11, 10,'2023-09-05', 'completed',  159.98),
  (12, 2, '2023-10-12', 'completed', 1299.99),
  (13, 3, '2023-11-20', 'shipped',    299.99),
  (14, 4, '2023-12-01', 'completed',   14.98),
  (15, 5, '2024-01-08', 'completed',  889.97);

-- Order items
CREATE TABLE IF NOT EXISTS order_items (
    item_id       INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL
);

INSERT OR IGNORE INTO order_items VALUES
  (1,  1, 1, 1, 1299.99),
  (2,  1, 2, 1,   29.99),
  (3,  2, 3, 1,  299.99),
  (4,  3, 4, 1,  499.99),
  (5,  3, 2, 1,   29.99),
  (6,  4, 7, 1,   89.99),
  (7,  5, 8, 1,  149.99),
  (8,  6,10, 1,   79.99),
  (9,  7, 9, 1,  379.99),
  (10, 8, 4, 1,  499.99),
  (11, 9, 2, 1,   29.99),
  (12,10, 1, 1, 1299.99),
  (13,10, 6, 2,    9.99),
  (14,11, 8, 1,  149.99),
  (15,11, 2, 1,    9.99),
  (16,12, 1, 1, 1299.99),
  (17,13, 3, 1,  299.99),
  (18,14, 5, 2,    4.99),
  (19,14, 6, 1,    9.99),
  (20,15, 1, 1, 1299.99);
"""


def seed_demo_database(connector: DatabaseConnector) -> None:
    """Seed the connected database with demo data if tables are not yet present."""
    existing = connector.get_table_names()
    if existing:
        logger.info("Database already has tables; skipping seed.")
        return
    logger.info("Seeding demo database …")
    with connector.engine.begin() as conn:
        for statement in _SEED_SQL.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt + ";"))
    logger.info("Demo database seeded successfully.")
