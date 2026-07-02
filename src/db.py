"""
============================================================
ML Feature Store -- PostgreSQL Database Layer
============================================================
Handles all database operations for storing and retrieving
fraud detection results.

What this does:
    - Connects to PostgreSQL using environment variables
    - Auto-creates the transactions table on first run
    - Provides functions to insert and query transaction results
    - Uses connection pooling (SimpleConnectionPool) for performance
    - All queries are parameterized to prevent SQL injection

Connection Pooling:
    Instead of a single persistent connection, we maintain a pool
    of 5-20 connections. Each operation borrows a connection, uses
    it, and returns it immediately. This eliminates per-query
    connection setup overhead and is the standard pattern used in
    production systems.

Environment Variables:
    DB_HOST     -- PostgreSQL host (default: localhost)
    DB_PORT     -- PostgreSQL port (default: 5432)
    DB_NAME     -- Database name (default: frauddb)
    DB_USER     -- Database user (default: frauduser)
    DB_PASSWORD -- Database password (default: fraudpass)
============================================================
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL connection pool and fraud transaction queries.

    Uses psycopg2.pool.SimpleConnectionPool to maintain a pool of
    reusable connections. This avoids the overhead of opening and
    closing a connection on every query, which is the primary
    bottleneck in single-connection setups.

    Usage:
        db = DatabaseManager()
        db.insert_transaction({...})
        recent = db.get_recent(limit=50)
    """

    # Pool configuration
    POOL_MIN_CONN = 5
    POOL_MAX_CONN = 20

    def __init__(self):
        """Initialize the connection pool using environment variables."""
        self.config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'dbname': os.getenv('DB_NAME', 'frauddb'),
            'user': os.getenv('DB_USER', 'frauduser'),
            'password': os.getenv('DB_PASSWORD', 'fraudpass'),
        }
        self._pool = None
        self._init_pool()
        self._create_table()

    def _init_pool(self):
        """
        Create the connection pool with retry logic.

        Retries 5 times with 3-second intervals because on startup
        PostgreSQL might not be ready yet (especially in Docker).
        """
        import time
        max_retries = 5
        retry_delay = 3

        for attempt in range(max_retries):
            try:
                self._pool = pool.SimpleConnectionPool(
                    minconn=self.POOL_MIN_CONN,
                    maxconn=self.POOL_MAX_CONN,
                    **self.config
                )
                logger.info(
                    f"Connection pool created ({self.POOL_MIN_CONN}-{self.POOL_MAX_CONN} connections) "
                    f"at {self.config['host']}:{self.config['port']}"
                )
                return
            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"PostgreSQL not ready (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to create connection pool after {max_retries} attempts")
                    raise

    @contextmanager
    def _get_conn(self):
        """
        Borrow a connection from the pool, yield it, then return it.

        This is the core pattern: every database operation should use
        this context manager. The connection is guaranteed to be
        returned to the pool even if an exception occurs.
        """
        conn = self._pool.getconn()
        conn.autocommit = True
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def _create_table(self):
        """
        Create the transactions table if it doesn't exist.

        Table Schema:
            id             -- Auto-incrementing primary key
            transaction_id -- UUID from the Kafka message
            amount         -- Transaction dollar amount
            is_fraud       -- Model's prediction (True/False)
            confidence     -- Model's confidence score (0.0 - 1.0)
            latency_ms     -- How long the prediction took in milliseconds
            source         -- Where the prediction came from ('api', 'batch', etc.)
            created_at     -- When this record was inserted
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            transaction_id UUID NOT NULL,
            amount FLOAT NOT NULL,
            is_fraud BOOLEAN NOT NULL,
            confidence FLOAT NOT NULL,
            latency_ms FLOAT NOT NULL,
            source VARCHAR(50) DEFAULT 'api',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Index on transaction_id for fast lookups (used by Redis cache checks)
        CREATE INDEX IF NOT EXISTS idx_transaction_id ON transactions(transaction_id);

        -- Index on created_at for time-range queries (used by Airflow retraining)
        CREATE INDEX IF NOT EXISTS idx_created_at ON transactions(created_at);
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
            logger.info("Transactions table ready")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise

    def insert_transaction(self, data: Dict) -> bool:
        """
        Insert a single fraud detection result into the database.

        Args:
            data: Dictionary with keys:
                - transaction_id (str/UUID)
                - amount (float)
                - is_fraud (bool)
                - confidence (float)
                - latency_ms (float)
                - source (str, optional)

        Returns:
            True if insert succeeded, False otherwise
        """
        insert_sql = """
        INSERT INTO transactions (transaction_id, amount, is_fraud, confidence, latency_ms, source)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (
                        data['transaction_id'],
                        data['amount'],
                        data['is_fraud'],
                        data['confidence'],
                        data['latency_ms'],
                        data.get('source', 'api')
                    ))
            return True
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return False

    def get_recent(self, limit: int = 100) -> List[Dict]:
        """
        Fetch the most recent N transactions.

        Used by the API to show recent activity and by Grafana
        dashboards to display real-time fraud detection results.
        """
        query = """
        SELECT transaction_id, amount, is_fraud, confidence, latency_ms, source, created_at
        FROM transactions
        ORDER BY created_at DESC
        LIMIT %s
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (limit,))
                    rows = cur.fetchall()
                    return [
                        {
                            **dict(row),
                            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                            'transaction_id': str(row['transaction_id'])
                        }
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_transactions_last_24hrs(self) -> List[Dict]:
        """
        Fetch all transactions from the last 24 hours.

        Used by the Airflow retraining DAG to pull recent data
        for nightly model retraining.
        """
        query = """
        SELECT transaction_id, amount, is_fraud, confidence, latency_ms, source, created_at
        FROM transactions
        WHERE created_at >= %s
        ORDER BY created_at DESC
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (cutoff,))
                    rows = cur.fetchall()
                    return [
                        {
                            **dict(row),
                            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                            'transaction_id': str(row['transaction_id'])
                        }
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_transaction_count(self) -> int:
        """Get total number of transactions in the database."""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM transactions")
                    return cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Count query failed: {e}")
            return 0

    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("PostgreSQL connection pool closed")
